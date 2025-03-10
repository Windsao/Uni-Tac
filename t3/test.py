import torch

from t3.models import T3
from t3.utils import logging
from t3.data_loader import WeightedDataLoader
from t3.models.nn_utils import mae_unpatchify, cross_mae_unpatchify, mae_unpatchify_pred_only, mae_apply_patchified_mask, get_device
import hydra
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm

from datetime import datetime
from .utils import is_main_process, get_entry_or, make_dataset_pie_plot
from .task_utils import rot_rmse, tra_rmse, count_classification_topk
from torchvision.transforms.v2 import ToPILImage

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize 


import os

try: 
    import wandb
except ImportError:
    wandb = None
    print("wandb is not installed, will not log to wandb")

import random
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class T3Test:
    def __init__(self, cfg, run_id=None):
        self.cfg = cfg
        
        self.model = None
        self.train_dataset = None
        self.eval_dataset = None
        self.img_preprocessors = None
        self.optimizer = None
        self.scheduler = None

        self.encoder_frozen = False
        self.trunk_frozen = False
        self.scheduled_unfreeze_step = -1

        self.min_avg_val_loss = np.inf

        self.device = get_device()
        
        if run_id is None:
            self.run_id = self.gen_run_id()
            if "comment" in self.cfg:
                self.run_id += "-" + self.cfg.comment
        else:
            self.run_id = run_id
        if self.cfg.train.wandb and wandb and is_main_process():
            wandb.init(
                project="TransferableTactileTransformer",
                config=OmegaConf.to_container(self.cfg, resolve=True),
                name=self.run_id,
                entity=self.cfg.train.wandb_entity)
            # define our custom x axis metric
            wandb.define_metric("train/step")
            wandb.define_metric("eval/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
    
    def gen_run_id(self):
        return f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"

    def setup_model(self):
        
        import pdb; pdb.set_trace()
        print('cfg', self.cfg)
        self.model = T3(self.cfg.network)
        self.encoder_frozen = False
        self.trunk_frozen = False
        self.scheduled_unfreeze_step = -1
        if get_entry_or(self.cfg.train, "freeze_encoder", False):
            self.model.freeze_encoder()
            self.encoder_frozen = True
            logging("Encoder will be frozen", True, "blue")
        if get_entry_or(self.cfg.train, "freeze_trunk", False):
            self.model.freeze_trunk()
            self.trunk_frozen = True
            logging("Trunk will be frozen", True, "blue")
        if self.encoder_frozen and self.trunk_frozen:
            if get_entry_or(self.cfg.train, "scheduled_unfreeze", False):
                self.scheduled_unfreeze_step = self.cfg.train.scheduled_unfreeze_step
                logging(f"Encoder and trunk will be frozen only until step {self.scheduled_unfreeze_step}", True, "blue")
        self.model.model_summary()
    
    def setup_optimizer(self):
        assert self.model is not None
        trunk_params = [v for k, v in self.model.named_parameters() if "trunk" in k]
        nontrunk_params = [v for k, v in self.model.named_parameters() if "trunk" not in k]
        params = [
            {"params": trunk_params},
            {"params": nontrunk_params, "lr": self.cfg.train.nontrunk_lr_scale * self.cfg.train.optimizer.lr}]
        self.optimizer = eval(self.cfg.train.optimizer["_target_"])(
            params=params,
            **{k: v for k, v in self.cfg.train.optimizer.items() if k != "_target_"})
        self.scheduler = hydra.utils.instantiate(self.cfg.train.scheduler, optimizer=self.optimizer)
    
    def setup_dataset(self):
        self.train_dataset = {}
        self.eval_dataset = {}
        
        # stats
        dataset_sizes = {}
        encoder_sizes = {}
        decoder_sizes = {}
        def _add_or_create_stat(d, key, value):
            if key.startswith("panda"):
                # combine all panda entries
                if "panda_probe" in d:
                    d["panda_probe"] += value
                else:
                    d["panda_probe"] = value
            elif key.startswith("cnc"):
                # combine all cnc entries
                if "cnc_probe" in d:
                    d["cnc_probe"] += value
                else:
                    d["cnc_probe"] = value
            else:
                if key in d:
                    d[key] += value
                else:
                    d[key] = value

        num_data_workers = self.cfg.train.num_data_workers

        def _get_dl_config(ds_cfg, folder, for_eval):
            res = ds_cfg.copy()
            res["data_dir"] = os.path.join(ds_cfg["data_dir"], folder)
            if for_eval:
                # turn off data augmentation for eval dataset
                res["random_resize_crop"] = False
                res["random_hv_flip_prob"] = 0
                res["color_jitter"] = None
            return res
        
        # load all datasets according to the config as one WeightedDataLoader
        for ds_name, ds_cfg in self.cfg.datasets.items():
            if ds_name.startswith("VAR_"):
                # skip the variables
                continue
            if not ds_cfg["activate"]:
                continue
            
            eval_only = ds_cfg["eval_only"]
            
            data_loader_cfg = dict(ds_cfg["data_loader"])
            data_loader_cfg["batch_size"] = self.cfg.train.batch_size

            train_ds_cfg = _get_dl_config(data_loader_cfg, "train", for_eval=eval_only)
            eval_ds_cfg = _get_dl_config(data_loader_cfg, "val", for_eval=True)
            
            train_ds = hydra.utils.instantiate(train_ds_cfg)
            eval_ds = hydra.utils.instantiate(eval_ds_cfg)

            self.eval_dataset[ds_name] = eval_ds.get_dataloader(num_data_workers)
            if eval_only:
                self.eval_dataset[f"{ds_name}_train"] = train_ds.get_dataloader(num_data_workers)
            else:
                self.train_dataset[ds_name] = train_ds.get_dataloader(num_data_workers)
            
            total_count = len(train_ds) * self.cfg.train.batch_size + len(eval_ds) * self.cfg.train.batch_size
            _add_or_create_stat(dataset_sizes, ds_name, total_count)
            _add_or_create_stat(encoder_sizes, data_loader_cfg["encoder_domain"], total_count)
            _add_or_create_stat(decoder_sizes, data_loader_cfg["decoder_domain"], total_count)
        self.train_dataloader = WeightedDataLoader(list(self.train_dataset.values()), weight_type=self.cfg.train.dl_weight_type)
        self.eval_dataloader = WeightedDataLoader(list(self.eval_dataset.values()), weight_type=self.cfg.train.dl_weight_type)
        logging(f"Total train batches: {len(self.train_dataloader)}, eval batches: {len(self.eval_dataloader)}", True, "blue")

        if self.cfg.train.wandb and wandb and is_main_process():
            # make dataset stat pie plots
            dataset_sizes_plot = make_dataset_pie_plot(dataset_sizes, "Dataset sizes", show=False)
            encoder_sizes_plot = make_dataset_pie_plot(encoder_sizes, "Encoder sizes", show=False)
            decoder_sizes_plot = make_dataset_pie_plot(decoder_sizes, "Decoder sizes", show=False)
            wandb.log({
                    f"stats/dataset_sizes": wandb.Image(dataset_sizes_plot),
                    f"stats/encoder_sizes": wandb.Image(encoder_sizes_plot),
                    f"stats/decoder_sizes": wandb.Image(decoder_sizes_plot),
                })

    @staticmethod
    def compose_loss_history(loss_history, enc_domain, dec_domain, loss, pred=None, Y=None, denormalize_func=None):
        # Add to all losses
        loss_history["all_losses"].append(loss.item())

        # add entry to loss_history
        entry_key = f"loss_{enc_domain}_{dec_domain}"
        if entry_key not in loss_history:
            loss_history[entry_key] = [loss.item()]
        else:
            loss_history[entry_key].append(loss.item())

        # RMSE for pose estimation
        if "pose_estimation_6d" in dec_domain and (pred is not None) and (Y is not None):
            rot_rmse_key = f"rot_rmse_{enc_domain}"
            tra_rmse_key = f"tra_rmse_{enc_domain}"

            rot_rmse_val = rot_rmse(pred, Y, denormalize_func=denormalize_func)
            tra_rmse_val = tra_rmse(pred, Y, denormalize_func=denormalize_func)

            if rot_rmse_key not in loss_history:
                loss_history[rot_rmse_key] = [rot_rmse_val]
            else: 
                loss_history[rot_rmse_key].append(rot_rmse_val)
            
            if tra_rmse_key not in loss_history:
                loss_history[tra_rmse_key] = [tra_rmse_val]
            else:
                loss_history[tra_rmse_key].append(tra_rmse_val)
        
        if "pose_estimation_3d" in dec_domain and (pred is not None) and (Y is not None):
            tra_rmse_key = f"tra_rmse_{enc_domain}"
            tra_rmse_val = tra_rmse(pred, Y, denormalize_func=denormalize_func)

            if tra_rmse_key not in loss_history:
                loss_history[tra_rmse_key] = [tra_rmse_val]
            else:
                loss_history[tra_rmse_key].append(tra_rmse_val)


        # classification accuracy
        if "cls" in dec_domain and (pred is not None) and (Y is not None):
            acc_top1_key = f"acc_{enc_domain}_top1"

            acc_top1_val = count_classification_topk(pred, Y, k=1) / len(Y)

            if acc_top1_key not in loss_history:
                loss_history[acc_top1_key] = [acc_top1_val]
            else:
                loss_history[acc_top1_key].append(acc_top1_val)
            


    
    def save_model(self, run_id, avg_val_loss, cur_step):
        if cur_step > 50 and avg_val_loss < self.min_avg_val_loss:
            # save as the best model
            self.min_avg_val_loss = avg_val_loss
            path = f"checkpoints/best_{run_id}"
            logging(f"Saving model to {path} as the best model", True, "green")
        else:
            path = f"checkpoints/{run_id}"

        logging(f"Current avg. test loss {avg_val_loss} v.s. best so far {self.min_avg_val_loss}. "\
                f"Saving model to {path}", True, "green")
        # save the model
        self.model.save_components(path)
        
        # save the optimizer and scheduler
        opt_type = self.cfg.train.optimizer["_target_"].split(".")[-1]
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer_{opt_type}.pt")
        sch_type = self.cfg.train.scheduler["_target_"].split(".")[-1]
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler_{sch_type}.pt")

        # save the config file
        with open(f"{path}/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
        # # save the git commit hash. Install gitpython and uncomment to use this feature
        # try:
        #     repo = git.Repo(search_parent_directories=True)
        #     with open(f"{path}/commit_hash.txt", "w") as f:
        #         f.write(repo.head.object.hexsha)
        #     del repo
        # except:
        #     logging("Failed to save git commit hash, ignored", True, "red")
    
    def load_model(self, path, load_optimizer=False, load_scheduler=False):
        # load the network
        self.model.load_components(path)
        logging(f"Loaded model from {path}", True, "green")
        self.model.to(self.device) # need to move the model to device before loading optimizer and scheduler

        # load the optimizer and scheduler
        if load_optimizer:
            opt_type = self.cfg.train.optimizer["_target_"].split(".")[-1]
            self.optimizer.load_state_dict(torch.load(f"{path}/optimizer_{opt_type}.pt"))
            logging(f"Loaded optimizer from {path}", True, "green")
        
        if load_scheduler:
            sch_type = self.cfg.train.scheduler["_target_"].split(".")[-1]
            self.scheduler.load_state_dict(torch.load(f"{path}/scheduler_{sch_type}.pt"))
            logging(f"Loaded scheduler from {path}", True, "green")
    
    def forward_once(self, data_batch):
        enc_domain = data_batch["encoder_domain"]
        dec_domain = data_batch["decoder_domain"]
        batch_x = data_batch["X"]

        # use label denormalize function to calculate RMSE
        if "pose_estimation_" in dec_domain:
            label_inv_normalize = data_batch["label_inv_normalize"]
        else:
            label_inv_normalize = None

        # set the domains & forward mode for the model
        if "electroassem" in dec_domain or "pose_estimation" in dec_domain:
            forward_mode = "multi_tower"
        else:
            forward_mode = "single_tower"
        self.model.set_domains(enc_domain, dec_domain, forward_mode)
        encoder = self.model.encoders[enc_domain] 
        if forward_mode == "single_tower":
            Xs = batch_x.to(self.device, non_blocking=True)
            # delete
            feature,_ ,_ = encoder(Xs)
            print('stop!!!!! here feature')
            print('feature', feature)
            print("Feature type:", type(feature)) 
            print('shape', feature.shape)
            print('stop!!!!! here feature---------------------------------------------------')
            pred = self.model(Xs)
        else:
            Xs = [x.to(self.device, non_blocking=True) for x in batch_x]
            feature,_ ,_ = [encoder(x) for x in Xs]
            print('stop!!!!! here feature')
            print('feature', feature)
            print("Feature type:", type(feature)) 
            print('shape', feature.shape)
            print('stop!!!!! here feature---------------------------------------------------')
            pred = self.model(*Xs)
        return ValueError('stop!!!!! here feature---------------------------------------------------')
        return label_inv_normalize, pred

    def train_test(self, run_id, total_train_steps, test_every, test_steps):
        self.model.to(self.device)
        cur_step = 0
        test_loss_history = self.test(test_steps, run_id, cur_step, 
                      enable_wandb=(self.cfg.train.wandb and wandb and is_main_process()))
    

    
    @torch.no_grad()
    def test(self, test_steps, run_id, cur_step, enable_wandb):
        self.model.to(self.device)
        test_iter = iter(self.eval_dataloader)
        self.model.eval()

        # 用于收集损失和评估指标
        losses = []
        test_loss_history = {"all_losses": [], "ssim": [], "psnr": []}

        # 从 config 里拿到 patch_size
        patch_size = self.cfg.network.patch_size

        pbar = tqdm(range(test_steps), position=0, leave=True)
        for idx in pbar:
            try:
                # 1. 取出一个 batch 数据
                data = next(test_iter)
            except StopIteration:
                # 如果数据集迭代完毕，重新创建迭代器
                test_iter = iter(self.eval_dataloader)
                data = next(test_iter)

            enc_domain = data["encoder_domain"]
            dec_domain = data["decoder_domain"]
            batch_y = data["Y"]

            inv_normalize_func = data["inv_normalize"]

            # 2. 前向推理
            label_inv_normalize, pred = self.forward_once(data)
            Y = batch_y.to(self.device)

            # 3. 计算 loss
            loss = self.model.compute_loss(pred, Y)
            losses.append(loss.item())

            # 4. 记录 loss 到字典
            self.compose_loss_history(
                test_loss_history,
                enc_domain,
                dec_domain,
                loss,
                pred,
                batch_y,
                denormalize_func=label_inv_normalize
            )

            # 5. 如果是 MAE，且需要可视化，生成一次可视化
            if get_entry_or(self.cfg.train, "generate_mae_visualizations", True) and idx == 0 and "mae" in dec_domain:
                (pred_imgs, mask, ids_restore) = pred
                self.generate_mae_visualizations(
                    data["X"].to(self.device, non_blocking=True),
                    patch_size,
                    pred_imgs,
                    mask,
                    inv_normalize_func,
                    run_id,
                    cur_step,
                )

            # 6. 如果是 MAE 任务，则计算并记录 SSIM/PSNR
            if "mae" in dec_domain:
                pred_imgs, mask, ids_restore = pred

                # 普通 MAE
                if pred_imgs.shape[1] == mask.shape[1]:
                    pred_unpatch = mae_unpatchify(pred_imgs, patch_size)
                else:
                    # cross MAE：需要用原图一起做 unpatchify
                    pred_unpatch = cross_mae_unpatchify(
                        pred_imgs,
                        data["X"].to(self.device),  # 原图（可放到GPU做unpatchify，然后再转CPU）
                        mask,
                        patch_size
                    )

                # 走一下 inv_normalize_func 后再转到 CPU
                pred_unpatch = inv_normalize_func(pred_unpatch).detach().cpu()
                ori_imgs = inv_normalize_func(data["X"]).detach().cpu()

                # 计算 SSIM 和 PSNR
                ssim_value, psnr_value = self.calculate_ssim_psnr(ori_imgs, pred_unpatch)
                test_loss_history["ssim"].append(ssim_value)
                test_loss_history["psnr"].append(psnr_value)

            if idx < 5:  # 可视化前5张图像
                self.visualize_differences(ori_imgs, pred_unpatch, indices=[0])

            # 7. 更新进度条信息
            avg_loss = np.mean(losses)
            avg_ssim = np.mean(test_loss_history["ssim"]) if test_loss_history["ssim"] else 0
            avg_psnr = np.mean(test_loss_history["psnr"]) if test_loss_history["psnr"] else 0
            pbar.set_description(
                f"Test {idx}/{test_steps} steps | Loss: {avg_loss:.4f}, SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f}"
            )

        # 8. 循环结束后，若开启 wandb，则记录总体统计
        if enable_wandb:
            log_items = {
                "eval/epoch": cur_step // len(self.train_dataloader),
                "eval/step": cur_step,
                "eval/avg_test_loss": avg_loss,
                "eval/avg_ssim": avg_ssim,
                "eval/avg_psnr": avg_psnr,
            }
            for k, v in test_loss_history.items():
                # ssim 和 psnr 已经单独记，不重复
                if k not in ["ssim", "psnr"]:
                    log_items[f"eval/{k}"] = np.mean(v)
            wandb.log(log_items)

        return test_loss_history




    def calculate_ssim_psnr(self, ori_imgs, pred_imgs):
        ssim_values = []
        psnr_values = []
        for idx, (ori_img, pred_img) in enumerate(zip(ori_imgs, pred_imgs)):
            # [3, H, W] -> [H, W, C]
            ori_img = ori_img.permute(1, 2, 0).cpu().numpy()
            pred_img = pred_img.permute(1, 2, 0).cpu().numpy()

            h, w, _ = ori_img.shape


            # # 计算缩放因子，确保最小边长 >= 7
            min_side = min(h, w)
            # if min_side < 7:
            #     scale_factor = 7 / min_side
            #     new_h = int(np.ceil(h * scale_factor))
            #     new_w = int(np.ceil(w * scale_factor))
            #     # 使用双线性插值进行图像放大
            #     ori_img = resize(ori_img, (new_h, new_w), mode='reflect', anti_aliasing=True)
            #     pred_img = resize(pred_img, (new_h, new_w), mode='reflect', anti_aliasing=True)
            #     h, w = new_h, new_w  # 更新尺寸

            # 动态调整窗口大小，确保不超过图像尺寸，且为奇数且 >= 3
            win_size = min(h, w, 7)
            if win_size % 2 == 0:
                win_size -= 1
            if win_size < 3:
                win_size = 3  # 最小窗口大小为3

            # 打印使用的窗口大小
            # logging(f"Image {idx} - Using win_size={win_size} for SSIM calculation", True, "blue")

            # 动态计算 data_range
            data_range = max(ori_img.max(), pred_img.max()) - min(ori_img.min(), pred_img.min())
            # logging(f"Image {idx} - Using data_range={data_range}", True, "blue")

            try:
                # 使用 channel_axis 替代 multichannel，并添加 data_range
                ssim_val = ssim(ori_img, pred_img, channel_axis=2, win_size=win_size, data_range=data_range)
                psnr_val = psnr(ori_img, pred_img, data_range=data_range)
                ssim_values.append(ssim_val)
                psnr_values.append(psnr_val)
                # logging(f"Image {idx} - SSIM: {ssim_val}, PSNR: {psnr_val}", True, "green")
            except ValueError as e:
                # 如果仍然出现问题，记录错误并跳过
                logging(f"SSIM/PSNR calculation failed for image {idx}: {e}", True, "red")
                continue

        # 如果所有图像都无法计算 SSIM/PSNR，返回0
        if len(ssim_values) == 0:
            return 0.0, 0.0

        return np.mean(ssim_values), np.mean(psnr_values)




    @torch.no_grad()
    def predict(self, enc_domain, dec_domain, batch_x):
        self.model.to(self.device)
        self.model.eval()
        # set the domains & forward mode for the model
        if "electroassem" in dec_domain or "pose_estimation" in dec_domain:
            forward_mode = "multi_tower"
        else:
            forward_mode = "single_tower"
        self.model.set_domains(enc_domain, dec_domain, forward_mode)

        if forward_mode == "single_tower":
            Xs = batch_x.to(self.device, non_blocking=True)
            pred = self.model(Xs)
        else:
            Xs = [x.to(self.device, non_blocking=True) for x in batch_x]
            pred = self.model(*Xs)
        return pred

    @torch.no_grad()
    def generate_mae_visualizations(self,
                                    imgs, patch_size, preds, masks, 
                                    inv_normalize_func, run_id, cur_step,
                                    num_to_generate=5,
                                    save=True):
        """
        unpatchify preds (N, L, patch_size**2 *3) back to images (N, 3, H, W)
        """
        if preds.shape[1] == masks.shape[1]:
            # the case for original MAE
            pred_imgs = inv_normalize_func(mae_unpatchify(preds, patch_size)).detach().cpu()
            pred_imgs_removed = inv_normalize_func(mae_unpatchify_pred_only(preds, imgs, masks, patch_size)).detach().cpu()
        else:
            # the case for cross MAE
            pred_imgs = inv_normalize_func(cross_mae_unpatchify(preds, imgs, masks, patch_size)).detach().cpu()
            pred_imgs_removed = pred_imgs
        ori_imgs = inv_normalize_func(imgs).detach().cpu()
        masked_imgs = mae_apply_patchified_mask(ori_imgs, masks, patch_size).detach().cpu()
        pil_converter = ToPILImage()
    
        imgs = []
        for i in range(min(num_to_generate, len(pred_imgs))):
            img = torch.cat([ori_imgs[i], masked_imgs[i], pred_imgs[i], pred_imgs_removed[i]], dim=2)
            imgs.append(img)
        # save all 5 images as one big image
        imgs = torch.cat(imgs, dim=1)
        pil_img = pil_converter(imgs)

        if save:
            # save images
            p = f"checkpoints/{run_id}/mae_visualizations/{cur_step}"
            os.makedirs(p, exist_ok=True)
            pil_img.save(os.path.join(p, f"visualize.jpg"))

        if self.cfg.train.wandb and wandb and is_main_process():
            wandb.log({
                    f"mae_visualizations/step": cur_step,
                    f"mae_visualizations/visualize": wandb.Image(pil_img),
                })
        return pil_img

    def train(self):
        
        if len(self.cfg.train.finetune_from) > 0:
            logging(f"WARNING: Loading existing model to finetune from {self.cfg.train.finetune_from}", True, "red")
            load_optimizer = get_entry_or(self.cfg.train, "load_optimizer", False)
            load_scheduler = get_entry_or(self.cfg.train, "load_scheduler", False)
            logging(f"Loading optimizer: {load_optimizer}, Loading scheduler: {load_scheduler}", True, "red")
            self.load_model(self.cfg.train.finetune_from, load_optimizer=load_optimizer, load_scheduler=load_scheduler)
        
        self.train_test(
            self.run_id, self.cfg.train.total_train_steps, self.cfg.train.test_every, self.cfg.train.test_steps)



    def visualize_differences(self, ori_imgs, pred_imgs, indices=[0, 1, 2]):
        import matplotlib.pyplot as plt

        def normalize_image(img):
            """将图像归一化到 [0, 1] 的范围"""
            img_min = img.min()
            img_max = img.max()
            return (img - img_min) / (img_max - img_min)

        for idx in indices:
            # 提取图像
            ori_img = ori_imgs[idx].permute(1, 2, 0).cpu().numpy()
            pred_img = pred_imgs[idx].permute(1, 2, 0).cpu().numpy()
            diff = np.abs(ori_img - pred_img)

            # 归一化图像到 [0, 1] 的范围
            ori_img = normalize_image(ori_img)
            pred_img = normalize_image(pred_img)
            diff = normalize_image(diff)

            # 可视化
            plt.figure(figsize=(18, 6))

            plt.subplot(1, 3, 1)
            plt.imshow(ori_img)
            plt.title("Original Image")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(pred_img)
            plt.title("Predicted Image")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(diff, cmap='hot')
            plt.title("Difference")
            plt.axis('off')

            plt.show()
            plt.savefig(f"visualization_diff_{idx}.png", bbox_inches='tight')

