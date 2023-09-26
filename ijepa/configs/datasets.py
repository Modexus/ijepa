
from hydra_zen import builds

from ijepa.datasets import load_imagenet

ImageNetConfig = builds(load_imagenet, split="train")
