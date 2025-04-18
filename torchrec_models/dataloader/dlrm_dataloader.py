#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List

from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DAYS,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.datasets.random import RandomRecDataset

STAGES = ["train", "val", "test"]

SINGLE_CAT_NAMES = ["t_cat_0"]
TEST_CAT_NAMES = ["t_cat_0", "t_cat_1", "t_cat_2", "t_cat_3"]
# TEST_CAT_NAMES = ["t_cat_0", "t_cat_1", "t_cat_2", "t_cat_3", "t_cat_4", "t_cat_5", "t_cat_6", "t_cat_7"]
AVAZU_CAT_NAMES = ["t_cat_0", "t_cat_1", "t_cat_2", "t_cat_3", "t_cat_4", "t_cat_5", "t_cat_6", "t_cat_7",
                    "t_cat_8", "t_cat_9", "t_cat_10", "t_cat_11", "t_cat_12", "t_cat_13", "t_cat_14", "t_cat_15",
                    "t_cat_16", "t_cat_17", "t_cat_18", "t_cat_19"]

def _get_random_dataloader(
    args: argparse.Namespace,
) -> DataLoader:

    return DataLoader(
        RandomRecDataset(
            keys=args.cat_name, # DEFAULT_CAT_NAMES
            batch_size=args.batch_size,
            hash_size=args.num_embeddings,
            hash_sizes=args.num_embeddings_per_feature
            if hasattr(args, "num_embeddings_per_feature")
            else None,
            manual_seed=args.seed if hasattr(args, "seed") else None,
            # ids_per_feature=args.hotness_list if args.hotness_list else 1,
            ids_per_feature=1,
            ids_per_features=args.hotness_list,
            num_dense=args.nDense,
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 0,
    )

def _get_single_dataloader(
    args: argparse.Namespace,
) -> DataLoader:
    return DataLoader(
        RandomRecDataset(
            keys=SINGLE_CAT_NAMES, # DEFAULT_CAT_NAMES
            batch_size=args.batch_size,
            hash_size=args.num_embeddings,
            hash_sizes=args.num_embeddings_per_feature
            if hasattr(args, "num_embeddings_per_feature")
            else None,
            manual_seed=args.seed if hasattr(args, "seed") else None,
            ids_per_feature=1,
            num_dense=len(SINGLE_CAT_NAMES),
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=args.pin_memory,
        num_workers=args.num_workers if hasattr(args, "num_workers") else 0,
    )


def _get_in_memory_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    files = os.listdir(args.in_memory_binary_criteo_path)

    def is_final_day(s: str) -> bool:
        return f"day_{DAYS - 1}" in s

    if stage == "train":
        # Train set gets all data except from the final day.
        files = list(filter(lambda s: not is_final_day(s), files))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        files = list(filter(is_final_day, files))
        rank = (
            dist.get_rank()
            if stage == "val"
            else dist.get_rank() + dist.get_world_size()
        )
        world_size = dist.get_world_size() * 2

    stage_files: List[List[str]] = [
        sorted(
            map(
                lambda x: os.path.join(args.in_memory_binary_criteo_path, x),
                filter(lambda s: kind in s, files),
            )
        )
        for kind in ["dense", "sparse", "labels"]
    ]
    dataloader = DataLoader(
        InMemoryBinaryCriteoIterDataPipe(
            *stage_files,  # pyre-ignore[6]
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_batches=args.shuffle_batches,
            hashes=args.num_embeddings_per_feature
            if args.num_embeddings is None
            else ([args.num_embeddings] * CAT_FEATURE_COUNT),
        ),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def get_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    """
    Gets desired dataloader from dlrm_main command line options. Currently, this
    function is able to return either a DataLoader wrapped around a RandomRecDataset or
    a Dataloader wrapped around an InMemoryBinaryCriteoIterDataPipe.

    Args:
        args (argparse.Namespace): Command line options supplied to dlrm_main.py's main
            function.
        backend (str): "nccl" or "gloo".
        stage (str): "train", "val", or "test".

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    args.pin_memory = (
        (backend == "nccl") if not hasattr(args, "pin_memory") else args.pin_memory
    )


    if args.single_table:
        # print("use random dataloader")
        return _get_single_dataloader(args)
    elif (
        not hasattr(args, "in_memory_binary_criteo_path")
        or args.in_memory_binary_criteo_path is None
    ):
        # print("use random dataloader")
        return _get_random_dataloader(args)
    else:
        # print("use criteo dataloader")
        return _get_in_memory_dataloader(args, stage)
