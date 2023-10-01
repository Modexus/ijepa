from functools import partial

from hydra_zen import builds
from torch import nn

from ijepa.models import VisionTransformer, VisionTransformerPredictor

VisionTransformerConf = builds(VisionTransformer, populate_full_signature=True)
VisionTransformerPredictorConf = builds(
    VisionTransformerPredictor,
    populate_full_signature=True,
)

vit_tiny_conf = VisionTransformerConf(
    embed_dim=192,
    num_heads=3,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

vit_small_conf = VisionTransformerConf(
    embed_dim=384,
    num_heads=6,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

vit_base_conf = VisionTransformerConf(
    embed_dim=768,
    num_heads=12,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

vit_large_conf = VisionTransformerConf(
    embed_dim=1024,
    num_heads=16,
    depth=24,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

vit_huge_conf = VisionTransformerConf(
    embed_dim=1280,
    num_heads=16,
    depth=32,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)

vit_giant_conf = VisionTransformerConf(
    embed_dim=1408,
    num_heads=16,
    depth=40,
    mlp_ratio=48 / 11,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
)
