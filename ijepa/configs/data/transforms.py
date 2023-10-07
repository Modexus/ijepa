from functools import partial

from hydra_zen import builds, just, store
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    PILToTensor,
    RandomResizedCrop,
)


def scale_01(image):
    return image / 255.0


PILToTensorConfig = builds(PILToTensor)
RandomResizedCropConfig = builds(
    RandomResizedCrop,
    size="${image_size}",
    scale=(0.3, 1.0),
    interpolation=InterpolationMode.BICUBIC,
    antialias=True,
)

BasicTransformsConfig = builds(
    Compose,
    transforms=[
        PILToTensorConfig,
        just(scale_01),
        RandomResizedCropConfig,
    ],
)

TinyImageNetTransformsConfig = builds(
    Compose,
    transforms=[
        PILToTensorConfig,
        just(scale_01),
        RandomResizedCropConfig(scale=(0.7, 1.0)),
    ],
)

CIFAR100TransformsConfig = builds(
    Compose,
    transforms=[
        PILToTensorConfig,
        just(scale_01),
        RandomResizedCropConfig,
    ],
)

transforms_store = store(group="transforms")

transforms_store(BasicTransformsConfig, name="basic")
transforms_store(TinyImageNetTransformsConfig, name="tinyimagenet")
transforms_store(CIFAR100TransformsConfig, name="cifar100")
