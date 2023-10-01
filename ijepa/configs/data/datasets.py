from hydra_zen import builds

from ijepa.datasets import load_imagenet_torch

from .transforms import BasicTransformsConfig

ImageNetConfig = builds(
    load_imagenet_torch, split="train", transforms=BasicTransformsConfig,
)
