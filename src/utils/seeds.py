import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import random

def set_seed(cfg=None):
    if cfg is None:
        import os
        from omegaconf import OmegaConf
        # Build path to configs/base.yaml robustly
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, "configs", "base.yaml")
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            seed = cfg.get("seed", 42)
        else:
            seed = 42
    elif isinstance(cfg, DictConfig) or hasattr(cfg, "seed"):
        seed = cfg.seed
    else:
        seed = int(cfg)
        
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)