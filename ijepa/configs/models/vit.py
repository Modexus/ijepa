from hydra_zen import builds, store
from torch import nn
from functools import partial
from ijepa.models import VisionTransformer, VisionTransformerPredictor

LayerNormConf = builds(nn.LayerNorm, zen_partial=True, populate_full_signature=True)
ViTEncoderBaseConf = builds(
    VisionTransformer,
    img_size=("${image_size}", "${image_size}"),
    patch_size="${patch_size}",
    in_chans=3,
    embed_dim=768,
    predictor_embed_dim=384,
    depth=12,
    predictor_depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    norm_layer=LayerNormConf(eps=1e-6),
    init_std=0.02,
    populate_full_signature=True,
)

ViTEncoderTinyConf = partial(ViTEncoderBaseConf, embed_dim=192, num_heads=3)
ViTEncoderSmallConf = partial(ViTEncoderBaseConf, embed_dim=384, num_heads=6)
ViTEncoderLargeConf = partial(
    ViTEncoderBaseConf, embed_dim=1024, num_heads=16, depth=24
)
ViTEncoderHugeConf = partial(ViTEncoderBaseConf, embed_dim=1280, num_heads=16, depth=32)
ViTEncoderGiantConf = partial(
    ViTEncoderBaseConf,
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

encoder_store = store(group="encoder")

encoder_store(ViTEncoderBaseConf, name="base")
encoder_store(ViTEncoderTinyConf, name="tiny")
encoder_store(ViTEncoderSmallConf, name="small")
encoder_store(ViTEncoderLargeConf, name="large")
encoder_store(ViTEncoderHugeConf, name="huge")
encoder_store(ViTEncoderGiantConf, name="giant")

predictor_store = store(group="predictor_partial")

predictor_store(ViTBasePredictorConf, name="base")
