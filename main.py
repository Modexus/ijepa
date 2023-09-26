# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
from pathlib import Path

import yaml
from loguru import logger

from ijepa import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname",
    type=str,
    help="name of config file to load",
    default="configs.yaml",
)
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0"],
    help="which devices to use on local machine",
)


def process_main(fname, devices):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[0].split(":")[-1])

    logger.info(f"called-params {fname}")

    # -- load script params
    params = None
    with Path(fname).open() as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)  #
        logger.info("loaded params...")

    app_main(args=params)


if __name__ == "__main__":
    args = parser.parse_args()

    process_main(args.fname, args.devices)
