from hydra_zen import builds, store
from torch.optim import AdamW

AdamWConf = builds(
    AdamW, weight_decay=0.4, zen_partial=True, populate_full_signature=True
)

optimizer_partial_store = store(group="optimizer_partial")
optimizer_partial_store(AdamWConf, name="adamw")
