# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from accelerate.utils import set_seed
from hydra_zen import store, zen

from ijepa import train
from ijepa.configs import ExperimentConfig

pre_seed = zen(lambda seed: set_seed(seed))
task_function = zen(train, pre_call=pre_seed)


if __name__ == "__main__":
    store(ExperimentConfig, name="train")
    store.add_to_hydra_store()

    task_function.hydra_main(
        config_name="train",
        version_base="1.1",
        config_path="configs",
    )
