from hydra_zen import builds, store
from torch.optim import AdamW

AdamWConf = builds(AdamW, zen_partial=True, populate_full_signature=True)

optimizer_store = store(package="optimizers")
optimizer_store(AdamWConf, name="adamw")
