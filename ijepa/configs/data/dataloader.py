from hydra_zen import builds, store
from torch.utils.data import DataLoader

from ijepa.configs.data.datasets import CIFAR100Conf, ImageNet1kConf, TinyImageNetConf
from ijepa.configs.masks import MultiBlockMaskBaseCollatorConf

TrainDataLoaderConfig = builds(
    DataLoader,
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=False,
    drop_last=True,
    batch_sampler=None,
)

CIFAR100TrainDataLoaderConf = builds(
    DataLoader,
    dataset=CIFAR100Conf,
    collate_fn=MultiBlockMaskBaseCollatorConf,
    builds_bases=(TrainDataLoaderConfig,),
)

ImageNet1kTrainDataLoaderConf = builds(
    DataLoader,
    dataset=ImageNet1kConf,
    collate_fn=MultiBlockMaskBaseCollatorConf,
    builds_bases=(TrainDataLoaderConfig,),
)

TinyImageNetTrainDataLoaderConf = builds(
    DataLoader,
    dataset=TinyImageNetConf,
    collate_fn=MultiBlockMaskBaseCollatorConf,
    builds_bases=(TrainDataLoaderConfig,),
)

data_loader_store = store(group="dataloader")

data_loader_store(ImageNet1kTrainDataLoaderConf, name="imagenet1k")
data_loader_store(TinyImageNetTrainDataLoaderConf, name="tinyimagenet")
data_loader_store(CIFAR100TrainDataLoaderConf, name="cifar100")
