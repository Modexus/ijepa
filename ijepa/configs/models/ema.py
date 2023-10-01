from hydra_zen import builds
from torch_ema import ExponentialMovingAverage

ExponentialMovingAverageBaseConf = builds(
    ExponentialMovingAverage,
    decay=0.995,
    zen_partial=True,
    populate_full_signature=True,
)
