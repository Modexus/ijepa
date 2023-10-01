from hydra_zen import builds
from transformers.optimization import get_cosine_schedule_with_warmup

CosineSchedulerBaseConf = builds(
    get_cosine_schedule_with_warmup,
    num_warmup_steps=40,
    zen_partial=True,
    populate_full_signature=True,
)
