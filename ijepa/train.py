from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers.training_utils import EMAModel
from hydra_zen.typing import Partial
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ijepa.masks.utils import apply_masks
from ijepa.utils.tensors import repeat_interleave_batch


def train(
    encoder: Module,
    predictor_partial: Partial[Module],
    dataloader: DataLoader,
    optimizer_partial: Partial[Optimizer],
    scheduler_partial: Partial[LRScheduler],
    momentum_scheduler_partial: Partial[EMAModel],
    num_epochs: int,
    base_lr: float,
    num_warmup_steps: int,
) -> None:
    accelerator = Accelerator(even_batches=False, log_with="wandb")
    accelerator.init_trackers("ijepa")

    predictor = predictor_partial(
        num_patches=encoder.patch_embed.num_patches,  # type: ignore
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
    )
    target_encoder = deepcopy(encoder)
    momentum_scheduler = momentum_scheduler_partial(encoder.parameters())

    effective_batch_size = (
        dataloader.batch_size  # type: ignore
        * accelerator.gradient_accumulation_steps
        * accelerator.num_processes
    )
    actual_lr = base_lr * effective_batch_size / 256
    optimizer = optimizer_partial(
        lr=actual_lr,
        params=list(encoder.parameters()) + list(predictor.parameters()),
    )
    actual_num_warmup_steps = min(1, num_warmup_steps * 256 // effective_batch_size)
    scheduler = scheduler_partial(
        optimizer,
        num_warmup_steps=actual_num_warmup_steps,
        num_training_steps=len(dataloader) * num_epochs,
    )

    encoder, predictor, target_encoder, momentum_scheduler = accelerator.prepare(
        encoder, predictor, target_encoder, momentum_scheduler
    )
    optimizer, dataloader, scheduler = accelerator.prepare(
        optimizer, dataloader, scheduler
    )

    for _ in tqdm(range(num_epochs)):
        for _, batch in tqdm(enumerate(dataloader)):
            batch = train_step(
                batch,
                encoder,
                target_encoder,
                predictor,
                scheduler,
                momentum_scheduler,
                optimizer,
                accelerator,
            )

    accelerator.end_training()


@torch.no_grad()
def forward_target(batch: dict[str, Any], target_encoder: Module) -> Tensor:
    target_encodings = target_encoder(batch["image"])
    target_encodings = F.layer_norm(
        target_encodings,
        (target_encodings.size(-1),),
    )  # normalize over feature-dim
    B = len(target_encodings)
    # -- create targets (masked regions of h)
    target_encodings = apply_masks(target_encodings, batch["masks_pred"])
    return repeat_interleave_batch(target_encodings, B, repeat=len(batch["masks_enc"]))


def forward_context(
    batch: dict[str, Any],
    encoder: Module,
    predictor: Module,
) -> Tensor:
    context_encodings = encoder(batch["image"], batch["masks_enc"])
    return predictor(context_encodings, batch["masks_enc"], batch["masks_pred"])


def loss_fn(input: Tensor, target: Tensor):  # noqa: A002
    return F.smooth_l1_loss(input, target)


def train_step(
    batch: dict[str, Any],
    encoder: Module,
    target_encoder: Module,
    predictor: Module,
    scheduler: LRScheduler,
    momentum_scheduler: EMAModel,
    optimizer: Optimizer,
    accelerator: Accelerator,
) -> dict[str, Any]:
    optimizer.zero_grad()
    # Step 1. Forward
    target_encodings = forward_target(batch, target_encoder)
    predicted_encodings = forward_context(batch, encoder, predictor)
    loss = loss_fn(predicted_encodings, target_encodings)

    #  Step 2. Backward & step
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()

    # Step 3. momentum update of target encoder
    momentum_scheduler.step(
        [param.cpu() for param in encoder.parameters()],  # type: ignore  # noqa: PGH003
    )

    batch["loss"] = loss.item()
    accelerator.log({"training_loss": batch["loss"]})
    accelerator.log({"lr": scheduler.get_last_lr()[0]})
    accelerator.log({"mask_enc_length": len(batch["masks_enc"][0][0])})
    accelerator.log({"mask_pred_length": len(batch["masks_pred"][0][0])})

    return batch
