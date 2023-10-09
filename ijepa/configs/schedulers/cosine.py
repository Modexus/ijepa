from hydra_zen import builds, store
from transformers.optimization import get_cosine_schedule_with_warmup

CosineSchedulerBaseConf = builds(
    get_cosine_schedule_with_warmup,
    zen_partial=True,
    populate_full_signature=True,
)

scheduler_partial_store = store(group="scheduler_partial")
scheduler_partial_store(CosineSchedulerBaseConf, name="cosine")
