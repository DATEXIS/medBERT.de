# from pytorch_lightning.loggers import WandbLogger
# from pytorch_lightning.plugins import DDPPlugin
from datamodule import MLMDataModule
from model import MLMModel
from pytorch_lightning.utilities.cli import LightningCLI

cli = LightningCLI(MLMModel, MLMDataModule, save_config_overwrite=True)
