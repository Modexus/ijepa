# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from loguru import logger

from ijepa.masks.utils import apply_masks
from ijepa.utils.logging import AverageMeter, gpu_timer, grad_logger
from ijepa.utils.tensors import repeat_interleave_batch


def train(
    encoder,
    predictor,
    dataloader,
    optimizer,
    scheduler,
    wd_scheduler,
    momentum_scheduler,
    num_epochs,
):
    target_encoder = deepcopy(encoder)

    accelerator = Accelerator()
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

        for _itr, (udata, masks_enc, masks_pred) in enumerate(dataloader):
            imgs = udata[0]
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        return repeat_interleave_batch(h, B, repeat=len(masks_enc))

                def forward_context():
                    z = encoder(imgs, masks_enc)
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
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(
                        encoder.parameters(),
                        target_encoder.parameters(),
                        strict=True,
                    ):
                        param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            assert not np.isnan(loss), "loss is nan"

        # -- Save Checkpoint after every epoch
        logger.info(f"avg. loss {loss_meter.avg:.3f}")


if __name__ == "__main__":
    train()
