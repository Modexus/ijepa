# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from accelerate.utils import set_seed
from hydra_zen import ZenStore, zen

from ijepa.configs import ExperimentConfig

pre_seed = zen(lambda seed: set_seed(seed))


def test(dataloader):
    5


task_function = zen(test, pre_call=pre_seed)


if __name__ == "__main__":
    store = ZenStore(deferred_hydra_store=False)
    store(ExperimentConfig, name="train")

    task_function.hydra_main(config_name="train", version_base="1.1", config_path=".")
