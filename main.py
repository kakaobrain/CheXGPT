import hydra
import lightning as pl
from omegaconf import OmegaConf, DictConfig
from labeler.datamodule.datamodule import CxrLabelerDataModule
from labeler.lightning.lightning import CxrLabelerLightningModule
from labeler.util.misc import update_config, set_env, get_logger


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = update_config(cfg)
    print(cfg)

    # Prepare training environment
    pl.seed_everything(42, workers=True)

    mode = cfg.mode
    set_env(cfg[mode].distributed)

    # Load a model
    ckpt_path = cfg[mode]["ckpt_path"]
    strict = cfg[mode].get("strict", True)
    lightningmodule = CxrLabelerLightningModule.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        cfg=cfg,
        map_location="cpu",
        strict=strict)

    # Build a datamodule
    tokenizer = lightningmodule.model.get_tokenizer()
    datamodule = CxrLabelerDataModule(tokenizer, cfg)

    # Run test/predict
    logger = get_logger(cfg.output.tensorboard_path)
    trainer = pl.Trainer(logger=logger, **cfg[mode].kwargs)

    if mode == "test":
        trainer.test(lightningmodule, datamodule=datamodule)
    elif mode == "predict":
        trainer.predict(lightningmodule, datamodule=datamodule)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")


if __name__ == "__main__":
    main()
