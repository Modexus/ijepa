from hydra_zen import builds
from torch.utils.data import DataLoader

from ijepa.configs.data.datasets import ImageNetConfig

TrainDataLoaderConfig = builds(
    DataLoader,
    batch_size=16,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
    drop_last=True,
)

ImageNetTrainDataLoaderConfig = builds(
    DataLoader,
    dataset=ImageNetConfig,
    builds_bases=(TrainDataLoaderConfig,),
)
