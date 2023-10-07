from hydra_zen import builds

from ijepa.masks import MBMaskCollator

MultiBlockMaskBaseCollatorConf = builds(
    MBMaskCollator,
    input_size=("${image_size}", "${image_size}"),
    enc_mask_scale=(0.85, 1.0),
    pred_mask_scale=(0.15, 0.2),
    aspect_ratio=(0.75, 1.5),
    nenc=1,
    npred=4,
    allow_overlap=False,
    populate_full_signature=True,
)
