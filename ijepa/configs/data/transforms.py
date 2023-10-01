from hydra_zen import builds, just
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
    size=224,
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
