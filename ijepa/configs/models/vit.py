from functools import partial

from hydra_zen import builds, store
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from ijepa.models import (
    VisionTransformer,
    VisionTransformerPredictor,
    ViT,
    ViTPredictor,
)

LayerNormConf = builds(nn.LayerNorm, zen_partial=True, populate_full_signature=True)
ViTEncoderBaseConf = builds(
    VisionTransformer,
    img_size=("${image_size}", "${image_size}"),
    patch_size="${patch_size}",
    in_chans=3,
    embed_dim=768,
    depth=12,
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

ViTEncoderTinyConf = builds(
    VisionTransformer, embed_dim=192, num_heads=3, builds_bases=(ViTEncoderBaseConf,)
)
ViTEncoderSmallConf = builds(ViTEncoderBaseConf, embed_dim=384, num_heads=6)
ViTEncoderLargeConf = builds(ViTEncoderBaseConf, embed_dim=1024, num_heads=16, depth=24)
ViTEncoderHugeConf = builds(ViTEncoderBaseConf, embed_dim=1280, num_heads=16, depth=32)
ViTEncoderGiantConf = builds(
    ViTEncoderBaseConf,
    embed_dim=1408,
    num_heads=16,
    depth=40,
    mlp_ratio=48 / 11,
)

ViTPredictorBaseConf = builds(
    VisionTransformerPredictor,
    populate_full_signature=True,
    zen_partial=True,
)

# torch
TransformerEncoderLayerBaseConf = builds(
    TransformerEncoderLayer,
    d_model=768,
    nhead=12,
    dim_feedforward=768 * 4,
    dropout=0.0,
    activation="gelu",
    layer_norm_eps=1e-6,
    batch_first=True,
    norm_first=True,
    populate_full_signature=True,
)

TransformerEncoderBaseConf = builds(
    TransformerEncoder,
    encoder_layer=TransformerEncoderLayerBaseConf,
    num_layers=12,
    populate_full_signature=True,
)

ViTEncoderTorchBaseConf = builds(
    ViT,
    image_size=("${image_size}", "${image_size}"),
    patch_size="${patch_size}",
    in_channels=3,
    embed_dim=768,
    encoder=TransformerEncoderBaseConf,
    populate_full_signature=True,
)

ViTEncoderTorchTinyConf = builds(
    ViT,
    embed_dim=192,
    encoder=TransformerEncoderBaseConf(
        encoder_layer=TransformerEncoderLayerBaseConf(
            d_model=192,
            nhead=3,
            dim_feedforward=192 * 4,
        )
    ),
    builds_bases=(ViTEncoderTorchBaseConf,),
)

ViTPredictorTorchBaseConf = builds(
    ViTPredictor,
    encoder_layer_partial=partial(
        TransformerEncoderLayerBaseConf,
        d_model=384,
        dim_feedforward=384 * 4,
    ),
    num_layers=6,
    predictor_embed_dim=384,
    populate_full_signature=True,
    norm_layer=LayerNormConf(eps=1e-6),
    zen_partial=True,
)

encoder_store = store(group="encoder")

encoder_store(ViTEncoderBaseConf, name="base")
encoder_store(ViTEncoderTinyConf, name="tiny")
encoder_store(ViTEncoderSmallConf, name="small")
encoder_store(ViTEncoderLargeConf, name="large")
encoder_store(ViTEncoderHugeConf, name="huge")
encoder_store(ViTEncoderGiantConf, name="giant")

encoder_store(ViTEncoderTorchBaseConf, name="torch_base")
encoder_store(ViTEncoderTorchTinyConf, name="torch_tiny")

predictor_store = store(group="predictor_partial")

predictor_store(ViTPredictorBaseConf, name="base")
predictor_store(ViTPredictorTorchBaseConf, name="torch_base")
