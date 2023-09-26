from math import pi

import pytorch_lightning as pl
import torch as tr
from hydra_zen import builds, make_config, make_custom_builds_fn
from torch.optim import AdamW
from torch.utils.data import DataLoader

pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=True)


ExperimentConfig = make_config(
    seed=1,
    data=None,
    mask=None,
    meta=None,
    optimization=None,
)

# ExperimentConfig = make_config(
#     seed=1,
#     lit_module=UniversalFuncModule,
#     trainer=builds(pl.Trainer, max_epochs=100),
#     model=builds(single_layer_nn, num_neurons=10),
#     optim=pbuilds(AdamW),
#     dataloader=pbuilds(DataLoader, batch_size=25, shuffle=True, drop_last=True),
#     target_fn=tr.cos,
#     training_domain=builds(tr.linspace, start=-2 * pi, end=2 * pi, steps=1000),
# )