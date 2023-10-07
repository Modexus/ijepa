from diffusers.training_utils import EMAModel
from hydra_zen import builds

EMAModelConf = builds(
    EMAModel,
    decay=0.995,
    zen_partial=True,
    populate_full_signature=False,
)
