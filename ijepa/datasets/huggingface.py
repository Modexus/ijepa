from typing import cast

from datasets import Dataset, load_dataset
from torch.nn import Module
from torchvision.transforms import Compose


def load_imagedataset_torch(
    name: str,
    split: str,
    transforms: Compose | Module,
) -> Dataset:
    ds = cast(Dataset, load_dataset(name, split=split))

    ds = ds.filter(
        lambda batch: [sample.mode == "RGB" for sample in batch],
        input_columns=["image"],
        batched=True,
        num_proc=10,
    )

    def transform_image(batch):
        batch["image"] = [transforms(img) for img in batch["image"]]
        return batch

    ds.set_transform(transform_image, columns=["image"])

    return ds
