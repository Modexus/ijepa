from hydra_zen import make_config

from ijepa.configs.data import ImageNetTrainDataLoaderConfig
from ijepa.configs.models import (
    ExponentialMovingAverageBaseConf,
    ViTBasePredictorConf,
    ViTEncoderTinyConf,
)
from ijepa.configs.optimizers import AdamWConf
from ijepa.configs.schedulers import CosineSchedulerBaseConf

ExperimentConfig = make_config(
    seed=42,
    encoder_partial=ViTEncoderTinyConf,
    predictor_partial=ViTBasePredictorConf,
    dataloader=ImageNetTrainDataLoaderConfig,
    optimizer_partial=AdamWConf,
    scheduler_partial=CosineSchedulerBaseConf,
    momentum_scheduler_partial=ExponentialMovingAverageBaseConf,
    num_epochs=10,
    image_size=224,
)
