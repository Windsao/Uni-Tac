import hydra
from t3.models import T3_Uni

@hydra.main(config_path="configs", config_name="uni_mae_encoder.yaml")
def main(cfg):
    model = T3_Uni(cfg)
    model.load_encoder()
    return model

if __name__ == "__main__":
    main()