from hydra_zen import make_config, make_custom_builds_fn

from ijepa.configs.data import ImageNetTrainDataLoaderConfig
from torch.utils.data import DataLoader

ExperimentConfig = make_config(
    seed=42,
    encoder=None,
    predictor=None,
    dataloader=ImageNetTrainDataLoaderConfig,
    optimizer=None,
    scheduler=None,
    wd_scheduler=None,
    momentum_scheduler=None,
    num_epochs=10,
)
