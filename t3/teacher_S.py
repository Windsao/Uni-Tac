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

import os
from t3.learn_utils import compute_encoder_distance


try: 
    import wandb
except ImportError:
    wandb = None
    print("wandb is not installed, will not log to wandb")

class T3Teacher:
    def __init__(self, cfg_teacher, cfg_student, run_id=None):
        self.cfg_teacher = cfg_teacher 
        self.cfg_student = cfg_student
        
        self.student_model = None
        self.teacher_model = None
        
        self.train_dataset = None
        self.eval_dataset = None
        self.img_preprocessors = None
        self.optimizer = None
        self.scheduler = None

        self.student_encoder_frozen = False
        self.student_trunk_frozen = False
        self.student_scheduled_unfreeze_step = -1

        self.min_avg_val_loss = np.inf

        self.device = get_device()
        
        if run_id is None:
            self.run_id = self.gen_run_id()
            if "comment" in self.cfg_student:
                self.run_id += "-" + self.cfg_student.comment
        else:
            self.run_id = run_id
        if self.cfg_student.train.wandb and wandb and is_main_process():
            wandb.init(
                project="TransferableTactileTransformer",
                config=OmegaConf.to_container(self.cfg_student, resolve=True),
                name=self.run_id,
                entity=self.cfg_student.train.wandb_entity)
            # define our custom x axis metric
            wandb.define_metric("train/step")
            wandb.define_metric("eval/step")
            # set all other train/ metrics to use this step
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("eval/*", step_metric="eval/step")
    
    def gen_run_id(self):
        return f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
    
    
    # change now
    def setup_model(self):
        # print(self.cfg_student.network)
        # print('---------------------------------------------------------------------------------------------')
        # print(self.cfg_teacher.network)
        # import pdb; pdb.set_trace()
        self.teacher_model = T3(self.cfg_teacher.network)
        self.student_model = T3(self.cfg_student.network)
        

        

        self.student_encoder_frozen = False
        self.student_trunk_frozen = False
        self.student_scheduled_unfreeze_step = -1
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False  # 教师模型的参数不参与梯度更新
            
        if get_entry_or(self.cfg_student.train, "freeze_encoder", False):
            self.student_model.freeze_encoder()
            self.student_encoder_frozen = True
            logging("Encoder will be frozen", True, "blue")
        if get_entry_or(self.cfg_student.train, "freeze_trunk", False):
            self.student_model.freeze_trunk()
            self.student_trunk_frozen = True
            logging("Trunk will be frozen", True, "blue")
        if self.student_encoder_frozen and self.student_trunk_frozen:
            if get_entry_or(self.cfg_student.train, "scheduled_unfreeze", False):
                self.student_scheduled_unfreeze_step = self.cfg_student.train.scheduled_unfreeze_step
                logging(f"Encoder and trunk will be frozen only until step {self.scheduled_unfreeze_step}", True, "blue")
        self.student_model.model_summary()
    
    
    def setup_optimizer(self):
        assert self.student_model is not None
        # student
        student_trunk_params = [v for k, v in self.student_model.named_parameters() if "trunk" in k]
        student_nontrunk_params = [v for k, v in self.student_model.named_parameters() if "trunk" not in k]
        # trunk_params = [v for k, v in self.model.named_parameters() if "trunk" in k]
        # nontrunk_params = [v for k, v in self.model.named_parameters() if "trunk" not in k]
        params = [
            {"params": student_trunk_params},
            {"params": student_nontrunk_params, "lr": self.cfg_student.train.nontrunk_lr_scale * self.cfg_student.train.optimizer.lr}]
        self.optimizer = eval(self.cfg_student.train.optimizer["_target_"])(
            params=params,
            **{k: v for k, v in self.cfg_student.train.optimizer.items() if k != "_target_"})
        self.scheduler = hydra.utils.instantiate(self.cfg_student.train.scheduler, optimizer=self.optimizer)
    
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

        num_data_workers = self.cfg_student.train.num_data_workers

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
        for ds_name, ds_cfg in self.cfg_student.datasets.items():
            if ds_name.startswith("VAR_"):
                # skip the variables
                continue
            if not ds_cfg["activate"]:
                continue
            
            eval_only = ds_cfg["eval_only"]
            
            data_loader_cfg = dict(ds_cfg["data_loader"])
            data_loader_cfg["batch_size"] = self.cfg_student.train.batch_size

            train_ds_cfg = _get_dl_config(data_loader_cfg, "train", for_eval=eval_only)
            eval_ds_cfg = _get_dl_config(data_loader_cfg, "val", for_eval=True)
            
            train_ds = hydra.utils.instantiate(train_ds_cfg)
            eval_ds = hydra.utils.instantiate(eval_ds_cfg)

            self.eval_dataset[ds_name] = eval_ds.get_dataloader(num_data_workers)
            if eval_only:
                self.eval_dataset[f"{ds_name}_train"] = train_ds.get_dataloader(num_data_workers)
            else:
                self.train_dataset[ds_name] = train_ds.get_dataloader(num_data_workers)
            
            total_count = len(train_ds) * self.cfg_student.train.batch_size + len(eval_ds) * self.cfg_student.train.batch_size
            _add_or_create_stat(dataset_sizes, ds_name, total_count)
            _add_or_create_stat(encoder_sizes, data_loader_cfg["encoder_domain"], total_count)
            _add_or_create_stat(decoder_sizes, data_loader_cfg["decoder_domain"], total_count)
        self.train_dataloader = WeightedDataLoader(list(self.train_dataset.values()), weight_type=self.cfg_student.train.dl_weight_type)
        self.eval_dataloader = WeightedDataLoader(list(self.eval_dataset.values()), weight_type=self.cfg_student.train.dl_weight_type)
        logging(f"Total train batches: {len(self.train_dataloader)}, eval batches: {len(self.eval_dataloader)}", True, "blue")

        if self.cfg_studentg.train.wandb and wandb and is_main_process():
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
    def compose_loss_history(loss_history, enc_domain, dec_domain, student_loss, student_pred=None,  Y=None, denormalize_func=None):
        # Add student loss to all losses
        loss_history["all_losses"].append(student_loss.item())
        
        # Add entry to loss_history for student model
        entry_key = f"loss_{enc_domain}_{dec_domain}_student"
        if entry_key not in loss_history:
            loss_history[entry_key] = [student_loss.item()]
        else:
            loss_history[entry_key].append(student_loss.item())


        # RMSE for pose estimation (student model)
        if "pose_estimation_6d" in dec_domain and student_pred is not None and Y is not None:
            rot_rmse_key = f"rot_rmse_{enc_domain}"
            tra_rmse_key = f"tra_rmse_{enc_domain}"

            rot_rmse_val = rot_rmse(student_pred, Y, denormalize_func=denormalize_func)
            tra_rmse_val = tra_rmse(student_pred, Y, denormalize_func=denormalize_func)

            if rot_rmse_key not in loss_history:
                loss_history[rot_rmse_key] = [rot_rmse_val]
            else: 
                loss_history[rot_rmse_key].append(rot_rmse_val)
            
            if tra_rmse_key not in loss_history:
                loss_history[tra_rmse_key] = [tra_rmse_val]
            else:
                loss_history[tra_rmse_key].append(tra_rmse_val)
                

            
        # Classification accuracy (student model)
        if "cls" in dec_domain and student_pred is not None and Y is not None:
            acc_top1_key = f"acc_student_{enc_domain}_top1"

            acc_top1_val = count_classification_topk(student_pred, Y, k=1) / len(Y)

            if acc_top1_key not in loss_history:
                loss_history[acc_top1_key] = [acc_top1_val]
            else:
                loss_history[acc_top1_key].append(acc_top1_val)




    @staticmethod
    def print_train_vs_test_stats(train_stat, test_stat):
        l = 35
        tl = 18
        logging("------- training vs eval stats -------", True, "blue")
        common_entries = set(train_stat.keys()).intersection(set(test_stat.keys()))
        for entry in sorted(common_entries):
            train_val = np.mean(train_stat[entry])
            test_val = np.mean(test_stat[entry])
            train_text = f"train: {train_val:.4f}".rjust(tl, ' ')
            val_text = f"test: {test_val:.4f}".rjust(tl, ' ')
            print(f"{entry.rjust(l, ' ')} \t {train_text} \t {val_text}")
        train_specific = set(train_stat.keys()).difference(common_entries)
        for entry in sorted(train_specific):
            train_val = np.mean(train_stat[entry])
            train_text = f"train: {train_val:.4f}".rjust(tl, ' ')
            print(f"{entry.rjust(l, ' ')} \t {train_text}")
        test_specific = set(test_stat.keys()).difference(common_entries)
        for entry in sorted(test_specific):
            test_val = np.mean(test_stat[entry])
            val_text = f"test: {test_val:.4f}".rjust(tl, ' ')
            print(f"{entry.rjust(l, ' ')} \t {' '*tl} \t {val_text}")
    
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
        self.student_model.save_components(path)
        
        # save the optimizer and scheduler
        opt_type = self.cfg_student.train.optimizer["_target_"].split(".")[-1]
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer_{opt_type}.pt")
        sch_type = self.cfg_student.train.scheduler["_target_"].split(".")[-1]
        torch.save(self.scheduler.state_dict(), f"{path}/scheduler_{sch_type}.pt")

        # save the config file
        with open(f"{path}/config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg_student))


    
    def load_model(self, path_student, path_teacher, load_optimizer=False, load_scheduler=False):
        # load the network
        self.student_model.load_components(path_student)
        logging(f"Loaded model from {path_student}", True, "green")
        self.student_model.to(self.device) # need to move the model to device before loading optimizer and scheduler

        # load the optimizer and scheduler
        if load_optimizer:
            opt_type = self.cfg_student.train.optimizer["_target_"].split(".")[-1]
            self.optimizer.load_state_dict(torch.load(f"{path_student}/optimizer_{opt_type}.pt"))
            logging(f"Loaded optimizer from {path_student}", True, "green")
        
        if load_scheduler:
            sch_type = self.cfg_student.train.scheduler["_target_"].split(".")[-1]
            self.scheduler.load_state_dict(torch.load(f"{path_student}/scheduler_{sch_type}.pt"))
            logging(f"Loaded scheduler from {path_student}", True, "green")
            
        # load the teacher model
        self.teacher_model.load_components(path_teacher)
        logging(f"Loaded teacher model from {path_teacher}", True, "green")
        self.teacher_model.to(self.device)

           

    
    # need to change  415
    def forward_once(self, data_batch, model="student"):
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

        if model == "teacher":
            # Teacher model: Select the encoder dynamically based on encoder_domain
            teacher_encoder = self.teacher_model.encoders[enc_domain]  # Assuming encoders are stored in the teacher model
            # teacher_features = teacher_encoder(batch_x.to(self.device, non_blocking=True))  # Forward through teacher encoder
            
            if forward_mode == "single_tower":
                Xs = batch_x.to(self.device, non_blocking=True)
                teacher_features = teacher_encoder(Xs)
                # pred = self.model(Xs)
            else:
                Xs = [x.to(self.device, non_blocking=True) for x in batch_x]
                teacher_features = [teacher_encoder(x) for x in Xs]  # 逐个编码
                # pred = self.model(*Xs)
                
            return teacher_features, label_inv_normalize, None

        elif model == "student":
            # Student model: Always use the fixed 'digit' encoder
            student_encoder = self.student_model.encoders['digit']  # Fixed encoder for student
            # student_features = student_encoder(batch_x.to(self.device, non_blocking=True))  # Forward through student encoder
            if forward_mode == "multi_tower":
                Xs = batch_x.to(self.device, non_blocking=True)
                student_features,_ ,_ = student_encoder(Xs)
                student_pred = self.student_model(*student_features)
            else:
                Xs = [x.to(self.device, non_blocking=True) for x in batch_x]
                student_features,_ ,_ = [student_encoder(x) for x in Xs]  # 逐个编码
                student_pred = self.student_model(student_features)
            return student_features, label_inv_normalize, student_pred
        else:
            raise ValueError("Model must be either 'student' or 'teacher'")

    def train_test(self, run_id, total_train_steps, test_every, test_steps):
        self.student_model.to(self.device)  # ss
        self.teacher_model.to(self.device)  # tt
        cur_step = 0
        train_iter = iter(self.train_dataloader)
        
        while cur_step < total_train_steps:
            # run training for test_every steps
            pbar = tqdm(range(test_every), position=0, leave=True)
            self.student_model.train()

            # unfreeze encoder and trunk if scheduled (student model)
            if self.student_scheduled_unfreeze_step > 0 and cur_step >= self.student_scheduled_unfreeze_step:
                if self.student_encoder_frozen:
                    self.student_model.unfreeze_encoder()
                    self.student_encoder_frozen = False
                    logging("Encoder unfrozen", True, "green")
                if self.student_trunk_frozen:
                    self.student_model.unfreeze_trunk()
                    self.student_trunk_frozen = False
                    logging("Trunk unfrozen", True, "green")

            train_loss_history = {"all_losses": []}
            for idx in pbar:
                cur_step += 1
                if cur_step >= total_train_steps:
                    break
                # step the dataloader
                data = next(train_iter)
                enc_domain = data["encoder_domain"]
                dec_domain = data["decoder_domain"]
                batch_y = data["Y"]
                
                self.optimizer.zero_grad()
                student_encoder, label_inv_normalize, student_pred = self.forward_once(data, model="student")
                # teacher forward pass
                with torch.no_grad():  # freeze teacher model
                    teacher_encoder, _, _= self.forward_once(data, model="teacher")

                Y = batch_y.to(self.device)
                student_loss = self.compute_loss(student_pred, Y)
                
                # encoder distance loss
                distance_loss = compute_encoder_distance(teacher_encoder, student_encoder, reduction="mean")

                # total loss
                total_loss = student_loss + 0.5* distance_loss
                
                total_loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # 记录损失
                self.compose_loss_history(train_loss_history, enc_domain, dec_domain, student_loss, student_pred=student_pred,  Y=Y, denormalize_func=label_inv_normalize)

                # logging, if enabled
                if self.cfg_student.train.wandb and wandb and is_main_process() and cur_step % self.cfg_student.train.log_freq == 1:
                    log_dict = {
                        f"train/loss_{enc_domain}_{dec_domain}": total_loss.item(),
                        f"train/epoch": cur_step // len(self.train_dataloader),
                        f"train/step": cur_step,
                        f"train/student_lr": self.optimizer.param_groups[0]["lr"]}
                    
                    # 可选的日志记录
                    if "pose_estimation_6d" in dec_domain:
                        log_dict[f"train/6dpe_rot_rmse_{enc_domain}"] = rot_rmse(student_pred, Y, denormalize_func=label_inv_normalize)
                        log_dict[f"train/6dpe_tra_rmse_{enc_domain}"] = tra_rmse(student_pred, Y, denormalize_func=label_inv_normalize)
                    if "cls" in dec_domain:
                        log_dict[f"train/acc_{dec_domain}_top1"] = count_classification_topk(student_pred.detach(), batch_y, k=1) / len(Y)
                    wandb.log(log_dict)
                pbar.set_description(f"Train {cur_step}/{total_train_steps} steps | loss: {total_loss.item():.4f}")
            
            # run eval for test_steps
            test_loss_history = self.test(test_steps, run_id, cur_step, enable_wandb=(self.cfg_student.train.wandb and wandb and is_main_process()))
            
            self.print_train_vs_test_stats(train_loss_history, test_loss_history)
            
            # save model
            if self.cfg_student.train.save_model and is_main_process():
                avg_val_loss = np.mean(test_loss_history["all_losses"])
                self.save_model(run_id, avg_val_loss, cur_step)
        return

    @torch.no_grad()
    def test(self, test_steps, run_id, cur_step, enable_wandb):
        self.student_model.to(self.device)
        test_iter = iter(self.eval_dataloader)
        self.student_model.eval()
        losses = []
        test_loss_history = {"all_losses": []}

        pbar = tqdm(range(test_steps), position=0, leave=True)
        for idx in pbar:
            # generate visualizations using inv_normalize_func
            data = next(test_iter)
            enc_domain = data["encoder_domain"]
            dec_domain = data["decoder_domain"]
            batch_y = data["Y"]
            # the denormalize function for the images
            inv_normalize_func = data["inv_normalize"]
            
            _, label_inv_normalize, pred = self.forward_once(data, model="student")
            Y = batch_y.to(self.device)

            loss = self.student_model.compute_loss(pred, Y)
            losses.append(loss.item())

            self.compose_loss_history(test_loss_history, enc_domain, dec_domain, loss, pred, batch_y, denormalize_func=label_inv_normalize)

            # obtain loss, and (optionally) generate mae visualizations
            if get_entry_or(self.cfg_student.train, "generate_mae_visualizations", True) and idx == 0 and "mae" in dec_domain:
                # generate visualizations
                (pred_imgs, mask, ids_restore) = pred
                self.generate_mae_visualizations(
                    data["X"].to(self.device, non_blocking=True), 
                    self.cfg_student.network.patch_size, pred_imgs, mask, inv_normalize_func, run_id, cur_step)
            
            pbar.set_description(
                f"Test {idx}/{test_steps} steps | loss: {loss.item():.4f}")
        if enable_wandb:
            log_items = {
                "eval/epoch": cur_step // len(self.train_dataloader),
                "eval/step": cur_step,
                f"eval/avg_test_loss": np.mean(losses)}
            for k, v in test_loss_history.items():
                log_items[f"eval/{k}"] = np.mean(v)
            wandb.log(log_items)

        return test_loss_history

    # @torch.no_grad()
    # def predict(self, enc_domain, dec_domain, batch_x):
    #     self.model.to(self.device)
    #     self.model.eval()
    #     # set the domains & forward mode for the model
    #     if "electroassem" in dec_domain or "pose_estimation" in dec_domain:
    #         forward_mode = "multi_tower"
    #     else:
    #         forward_mode = "single_tower"
    #     self.model.set_domains(enc_domain, dec_domain, forward_mode)

    #     if forward_mode == "single_tower":
    #         Xs = batch_x.to(self.device, non_blocking=True)
    #         pred = self.model(Xs)
    #     else:
    #         Xs = [x.to(self.device, non_blocking=True) for x in batch_x]
    #         pred = self.model(*Xs)
    #     return pred

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

        if self.cfg_student.train.wandb and wandb and is_main_process():
            wandb.log({
                    f"mae_visualizations/step": cur_step,
                    f"mae_visualizations/visualize": wandb.Image(pil_img),
                })
        return pil_img

    def train(self):
        
        if len(self.cfg_student.train.finetune_from) > 0:
            logging(f"WARNING: Loading existing model to finetune from {self.cfg_student.train.finetune_from}", True, "red")
            load_optimizer = get_entry_or(self.cfg_student.train, "load_optimizer", False)
            load_scheduler = get_entry_or(self.cfg_student.train, "load_scheduler", False)
            logging(f"Loading optimizer: {load_optimizer}, Loading scheduler: {load_scheduler}", True, "red")
            self.load_model(self.cfg_student.train.finetune_from, self.cfg_teacher.train.finetune_from, load_optimizer=load_optimizer, load_scheduler=load_scheduler)
        
        self.train_test(
            self.run_id, self.cfg_student.train.total_train_steps, self.cfg_student.train.test_every, self.cfg_student.train.test_steps)
