import hydra

from t3 import T3Pretrain
# from t3 import T3Test

@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def train_nn(cfg):
    pretrainer = T3Pretrain(cfg, run_id="test")
    # pretrainer = T3Test(cfg)
    pretrainer.setup_model()
    pretrainer.setup_optimizer()
    pretrainer.setup_dataset()
    print("Dataset setup complete")
    pretrainer.train()
    pretrainer.test(1, "", 0, False)

if __name__ == "__main__":
    train_nn()