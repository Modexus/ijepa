from hydra_zen import builds, just
from torchvision.transforms.v2 import (
    Compose,
    InterpolationMode,
    PILToTensor,
    RandomResizedCrop,
)

PILToTensorConfig = builds(PILToTensor)


def scale(image):
    return image / 255.0


RandomResizedCropConfig = builds(
    RandomResizedCrop,
    size=224,
    scale=(0.3, 1.0),
    interpolation=InterpolationMode.BICUBIC,
)

BasicTransformsConfig = builds(
    Compose,
    transforms=[
        PILToTensorConfig,
        just(scale),
        RandomResizedCropConfig,
    ],
)
