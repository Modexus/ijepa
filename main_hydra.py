# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import pytorch_lightning as pl
from hydra_zen import ZenStore, to_yaml, zen

from ijepa import train
from ijepa.configs import ExperimentConfig, ImageNetConfig

pre_seed = zen(lambda seed: pl.seed_everything(seed))

task_function = zen(lambda cfg: print(to_yaml(cfg)))

# task_function = zen(train, pre_call=pre_seed)

if __name__ == "__main__":
    store = ZenStore(deferred_hydra_store=False)
    store(ImageNetConfig, name="train")


    task_function.hydra_main(
        config_name="train", version_base="1.1", config_path="."
    )
