from hydra_zen import builds, store

from ijepa.datasets import load_imagedataset_torch

from .transforms import (
    BasicTransformsConfig,
    CIFAR100TransformsConfig,
    TinyImageNetTransformsConfig,
)

ImageNet1kConf = builds(
    load_imagedataset_torch,
    name="imagenet-1k",
    split="train",
    transforms=BasicTransformsConfig,
)

TinyImageNetConf = builds(
    load_imagedataset_torch,
    name="zh-plus/tiny-imagenet",
    split="train",
    transforms=TinyImageNetTransformsConfig,
)

CIFAR100Conf = builds(
    load_imagedataset_torch,
    name="cifar100",
    split="train",
    transforms=CIFAR100TransformsConfig,
)

datasets_store = store(group="datasets")

datasets_store(ImageNet1kConf, name="imagenet1k")
datasets_store(TinyImageNetConf, name="tinyimagenet")
datasets_store(CIFAR100Conf, name="cifar100")
