from hydra_zen import make_config, store

from ijepa.configs.data import (
    ImageNet1kTrainDataLoaderConf,
    TinyImageNetTrainDataLoaderConf,
)
from ijepa.configs.models import (
    EMAModelConf,
    ViTEncoderTinyConf,
    ViTEncoderTorchTinyConf,
    ViTPredictorBaseConf,
    ViTPredictorTorchBaseConf,
)
from ijepa.configs.optimizers import AdamWConf
from ijepa.configs.schedulers import CosineSchedulerBaseConf

TrainImagenet1kConf = make_config(
    seed=42,
    num_epochs=10,
    image_size=224,
    patch_size=16,
    base_lr=1e-3,
    num_warmup_steps=40,
    encoder=ViTEncoderTinyConf,
    predictor_partial=ViTPredictorBaseConf,
    dataloader=ImageNet1kTrainDataLoaderConf,
    optimizer_partial=AdamWConf,
    scheduler_partial=CosineSchedulerBaseConf,
    momentum_scheduler_partial=EMAModelConf,
)

TrainTinyImageNetConf = make_config(
    image_size=56,
    patch_size=4,
    encoder=ViTEncoderTorchTinyConf,
    predictor_partial=ViTPredictorTorchBaseConf,
    dataloader=TinyImageNetTrainDataLoaderConf(batch_size=128),
    bases=(TrainImagenet1kConf,),
)

store(TrainImagenet1kConf, name="train_imagenet1k")
store(TrainTinyImageNetConf, name="train_tinyimagenet")
