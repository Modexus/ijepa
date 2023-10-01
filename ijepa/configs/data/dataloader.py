from hydra_zen import builds
from torch.utils.data import DataLoader

from ijepa.configs.data.datasets import ImageNetConfig
from ijepa.configs.masks import MultiBlockMaskBaseCollatorConf

TrainDataLoaderConfig = builds(
    DataLoader,
    batch_size=16,
    shuffle=True,
    num_workers=0,  # 10,
    pin_memory=False,
    drop_last=True,
    batch_sampler=None,
)

ImageNetTrainDataLoaderConfig = builds(
    DataLoader,
    dataset=ImageNetConfig,
    collate_fn=MultiBlockMaskBaseCollatorConf,
    builds_bases=(TrainDataLoaderConfig,),
)
