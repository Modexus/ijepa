from datasets import Dataset, load_dataset


def load_imagenet(split: str = "train") -> Dataset:
    ds = load_dataset("imagenet-1k", split=split)
    ds.set_format("torch")

    return ds