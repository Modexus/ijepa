from hydra_zen import builds, store
from torch import nn

from ijepa.models import VisionTransformer, VisionTransformerPredictor

LayerNormConf = builds(nn.LayerNorm, zen_partial=True, populate_full_signature=True)
ViTEncoderBaseConf = builds(
    VisionTransformer,
    norm_layer=LayerNormConf(eps=1e-6),
    zen_partial=True,
    populate_full_signature=True,
)

ViTEncoderTinyConf = ViTEncoderBaseConf(
    embed_dim=192,
    num_heads=3,
)

ViTEncoderSmallConf = ViTEncoderBaseConf(
    embed_dim=384,
    num_heads=6,
)

ViTEncoderLargeConf = ViTEncoderBaseConf(
    embed_dim=1024,
    num_heads=16,
    depth=24,
)

ViTEncoderHugeConf = ViTEncoderBaseConf(
    embed_dim=1280,
    num_heads=16,
    depth=32,
)

ViTEncoderGiantConf = ViTEncoderBaseConf(
    embed_dim=1408,
    num_heads=16,
    depth=40,
    mlp_ratio=48 / 11,
)

ViTBasePredictorConf = builds(
    VisionTransformerPredictor,
    populate_full_signature=True,
    zen_partial=True,
)

model_store = store(package="models")
encoder_store = model_store(group="encoder")

ViTBase = encoder_store(ViTEncoderBaseConf, name="base")
encoder_store(ViTEncoderTinyConf, name="tiny")
encoder_store(ViTEncoderSmallConf, name="small")
encoder_store(ViTEncoderLargeConf, name="large")
encoder_store(ViTEncoderHugeConf, name="huge")
encoder_store(ViTEncoderGiantConf, name="giant")

predictor_store = model_store(group="predictor")

predictor_store(ViTBasePredictorConf, name="base")
