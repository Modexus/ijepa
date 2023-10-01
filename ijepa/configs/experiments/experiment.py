from hydra_zen import make_config

from ijepa.configs.data import ImageNetTrainDataLoaderConfig
from ijepa.configs.models import ViTBasePredictorConf, ViTEncoderTinyConf

ExperimentConfig = make_config(
    seed=42,
    encoder=ViTEncoderTinyConf,
    predictor=ViTBasePredictorConf,
    dataloader=ImageNetTrainDataLoaderConfig,
    optimizer=None,
    scheduler=None,
    wd_scheduler=None,
    momentum_scheduler=None,
    num_epochs=10,
    image_size=224,
)
