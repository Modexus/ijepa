from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from hydra_zen.typing import Partial
from loguru import logger
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from ijepa.masks.utils import apply_masks
from ijepa.utils.logging import AverageMeter, gpu_timer, grad_logger
from ijepa.utils.tensors import repeat_interleave_batch


def train(
    encoder_partial: Partial[Module],
    predictor_partial: Partial[Module],
    dataloader: DataLoader,
    optimizer_partial: Partial[Optimizer],
    scheduler_partial: Partial[LRScheduler],
    momentum_scheduler_partial: ExponentialMovingAverage,
    num_epochs: int,
    image_size: int,
):
    encoder = encoder_partial(img_size=(image_size, image_size))
    predictor = predictor_partial(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
    )

    target_encoder = deepcopy(encoder)
    momentum_scheduler = momentum_scheduler_partial(encoder.parameters())

    optimizer = optimizer_partial(
        params=list(encoder.parameters()) + list(predictor.parameters()),
    )
    scheduler = scheduler_partial(
        optimizer,
        num_training_steps=num_epochs * len(dataloader),
    )

    accelerator = Accelerator(even_batches=False)
    (
        encoder,
        predictor,
        target_encoder,
        optimizer,
        dataloader,
        scheduler,
    ) = accelerator.prepare(
        encoder,
        predictor,
        target_encoder,
        optimizer,
        dataloader,
        scheduler,
    )

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}")

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for batch in iter(dataloader):
            images = batch["images"]
            masks_enc = batch["masks_enc"]
            masks_pred = batch["masks_pred"]
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(images)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        return repeat_interleave_batch(h, B, repeat=len(masks_enc))

                def forward_context():
                    z = encoder(images, masks_enc)
                    return predictor(z, masks_enc, masks_pred)

                def loss_fn(z, h):
                    return F.smooth_l1_loss(z, h)

                # Step 1. Forward
                h = forward_target()
                z = forward_context()
                loss = loss_fn(z, h)

                #  Step 2. Backward & step
                accelerator.backward(loss)
                optimizer.step()

                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    momentum_scheduler.update()

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint after every epoch
        logger.info(f"avg. loss {loss_meter.avg:.3f}")


if __name__ == "__main__":
    train()
