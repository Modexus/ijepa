from datasets import Dataset, load_dataset
from torch.nn import Module
from torchvision.transforms import PILToTensor


def load_imagenet_torch(
    split: str = "train",
    transforms: Module = PILToTensor,
) -> Dataset:
    ds = load_dataset("imagenet-1k", split=split)

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


def load_cifar100_torch(
    split: str = "train",
    transforms: Module = PILToTensor,
) -> Dataset:
    ds = load_dataset("cifar100", split=split)

    def transform_image(batch):
        batch["image"] = [transforms(img) for img in batch["image"]]
        return batch

    ds.set_transform(transform_image, columns=["image"])

    return ds
