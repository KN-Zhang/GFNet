import os
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.distributed import DistributedDataParallel
from loguru import logger
import gc

import configs

class CheckPoint:
    def __init__(self, dir=None, name="tmp"):
        self.name = name
        self.dir = dir
        os.makedirs(os.path.join(self.dir, self.name), exist_ok=True)

    def save(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        ):
        if configs.cfg.RANK == 0:
            assert model is not None
            if isinstance(model, (DataParallel, DistributedDataParallel)):
                model = model.module
            states = {
                "model": model.state_dict(),
                "n": n,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
            torch.save(states, os.path.join(self.dir, self.name, "latest.pth"))
            logger.info(f"Saved states {list(states.keys())}, at step {n}")
    
    def load(
        self,
        model,
        optimizer,
        lr_scheduler,
        n,
        ):   
        if os.path.exists(os.path.join(self.dir, self.name, "latest.pth")) and configs.cfg.RANK == 0:
            states = torch.load(os.path.join(self.dir, self.name, "latest.pth"))
            if "model" in states:
                model.load_state_dict(states["model"])
            if "n" in states:
                n = states["n"] if states["n"] else n
            if "optimizer" in states:
                try:
                    optimizer.load_state_dict(states["optimizer"])
                except Exception as e:
                    print(f"Failed to load states for optimizer, with error {e}")
            if "lr_scheduler" in states:
                lr_scheduler.load_state_dict(states["lr_scheduler"])
            print(f"Loaded states {list(states.keys())}, at step {n}")
            del states
            gc.collect()
            torch.cuda.empty_cache()

        return model, optimizer, lr_scheduler, n