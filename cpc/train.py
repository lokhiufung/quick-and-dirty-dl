import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf

from cpc.model import CPCAudioRawModel


@hydra.main(config_name='config/cpc_config.yaml')
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    model_ckpt_callback = ModelCheckpoint(monitor='val_loss')
    trainer = pl.Trainer(**cfg.trainer, callbacks=[model_ckpt_callback])
    cpc_model = CPCAudioRawModel(cfg=DictConfig(cfg.model))

    trainer.fit(cpc_model)


if __name__ == '__main__':
    main()