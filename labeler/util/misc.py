import os
from lightning.pytorch.loggers import TensorBoardLogger


def update_config(cfg):
    # Set prefetch_factor to None, if num_workers is 0
    if cfg.dataloader.test.num_workers == 0:
        cfg.dataloader.test.prefetch_factor = None
    if cfg.dataloader.predict.num_workers == 0:
        cfg.dataloader.predict.prefetch_factor = None

    # Set global WORLD_SIZE (the number of total gpus)
    cfg.test.distributed.WORLD_SIZE = cfg.test.distributed.NUM_GPUS_PER_NODE * cfg.test.distributed.NUM_NODES
    cfg.predict.distributed.WORLD_SIZE = cfg.predict.distributed.NUM_GPUS_PER_NODE * cfg.predict.distributed.NUM_NODES

    # Set per-gpu batch size
    cfg.dataloader.test.batch_size = cfg.dataloader.test.batch_size // cfg.test.distributed.WORLD_SIZE
    cfg.dataloader.predict.batch_size = cfg.dataloader.predict.batch_size // cfg.predict.distributed.WORLD_SIZE

    return cfg


def set_env(distribued_cfg):
    for k, v in distribued_cfg.items():
        os.environ[k] = str(v)


def get_logger(tensorboard_path):
    tensorboard_logger = TensorBoardLogger(
        save_dir=tensorboard_path,
        name="",
    )
    return tensorboard_logger
