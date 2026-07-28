import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="base", version_base=None)
def set_seed(cfg: DictConfig):
    print("Seed is set:", cfg.seed)

print("Before set_seed")
set_seed()
print("After set_seed - DOES THIS PRINT?")
