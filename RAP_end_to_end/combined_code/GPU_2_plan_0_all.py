import argparse
import itertools
import os
import sys
sys.path.append('/workspace/RAP/torchrec_models')

from typing import Iterator, List

import torch
import torch.optim as optim
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import KeyedOptimizerWrapper
import torch.multiprocessing as mp

from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.planner.types import ParameterConstraints
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from dlrm_models import DLRM_MLP
from dataloader.dlrm_dataloader import get_dataloader
from dlrm_parser import get_args
from torchrec_utils import *
from sharded_embedding_table import ShardedEmbeddingTable
from torch.utils.cpp_extension import load
import cudf
from torchrec_prepare_data import generate_parquet_file_based_on_mapping
import random
import threading

from torchrec.distributed.shard import shard
from torchrec.distributed.types import (
    ParameterSharding,
    ShardingPlan,
    EnumerableShardingSpec,
    ShardMetadata,
    ShardingEnv,
    ShardingType
)
from tqdm import tqdm
import time

from torchrec.distributed.dist_data import (
    KJTAllToAll,
    PooledEmbeddingsAllToAll,
)

import warnings

def free_memory(a, b, c):
    del a
    del b
    del c

def queue_wait(queue):
    while queue.empty():
        time.sleep(0.000001)
    a = queue.get()

def training_iter_without_input_dist(queue_list, sparse_optimizer, dense_optimizer, loss_fn, dist_input, dense_input, label_tensor, sharded_emts, mlp_layers, rank, work_group):
    sparse_optimizer.zero_grad()
    dense_optimizer.zero_grad()

    
    sparse_feature = sharded_emts.forward_on_dist_input(dist_input) # forward

    
    logits = mlp_layers(dense_input, sparse_feature)
    
    
    loss = loss_fn(logits, label_tensor)
    loss.backward()

    queue_list[rank].put(1)
    average_gradients(mlp_layers, work_group)
    
    sparse_optimizer.step()
    dense_optimizer.step()


def thread_cpu_part_gpu_0_next(df_dense, df_sparse, cuda_preprocess, this_rank, device, args, input_list, output_list, dup_times):
    # ==================== input_list =====================
    fill_null_dense_output_tensor_list_0, fill_null_dense_output_tensor_list_2, fill_null_sparse_output_tensor_list_3, x_list_5, border_list_6 = input_list

    cuda_preprocess.init_cuda(this_rank)  # set cuda device
    # ==================== input_list + cpu code =====================
    for _ in range(dup_times):
        # fill_null_dense-0 input_list_code:
        fill_null_dense_ptr_list_0 = [df_dense["int_0"].data.ptr, df_dense["int_1"].data.ptr, df_dense["int_2"].data.ptr, df_dense["int_3"].data.ptr, df_dense["int_4"].data.ptr, df_dense["int_5"].data.ptr, df_dense["int_6"].data.ptr, df_dense["int_7"].data.ptr, df_dense["int_8"].data.ptr, df_dense["int_9"].data.ptr, df_dense["int_10"].data.ptr, df_dense["int_11"].data.ptr, df_dense["int_12"].data.ptr]
        # fill_null_dense-0 cpu_part_code:
        fill_null_dense_ptr_list_0_tensor = torch.tensor(fill_null_dense_ptr_list_0, dtype=torch.int64, device=device)
        # logit-1 input_list_code:
        logit_input_list_1 = [fill_null_dense_output_tensor_list_0[0], fill_null_dense_output_tensor_list_0[1], fill_null_dense_output_tensor_list_0[2], fill_null_dense_output_tensor_list_0[3], fill_null_dense_output_tensor_list_0[4], fill_null_dense_output_tensor_list_0[5], fill_null_dense_output_tensor_list_0[6], fill_null_dense_output_tensor_list_0[7], fill_null_dense_output_tensor_list_0[8], fill_null_dense_output_tensor_list_0[9], fill_null_dense_output_tensor_list_0[10], fill_null_dense_output_tensor_list_0[11], fill_null_dense_output_tensor_list_0[12]]
        # logit-1 cpu_part_code:
        logit_input_ptr_1, logit_input_length_1 = cuda_preprocess.logit_cpu_part_base(logit_input_list_1)
        # fill_null_dense-2 input_list_code:
        fill_null_dense_ptr_list_2 = [df_sparse["int_0"].data.ptr]
        # fill_null_dense-2 cpu_part_code:
        fill_null_dense_ptr_list_2_tensor = torch.tensor(fill_null_dense_ptr_list_2, dtype=torch.int64, device=device)
        # fill_null_sparse-3 input_list_code:
        fill_null_sparse_ptr_list_3 = [df_sparse["cat_0"].data.ptr, df_sparse["cat_3"].data.ptr, df_sparse["cat_4"].data.ptr, df_sparse["cat_5"].data.ptr, df_sparse["cat_12"].data.ptr, df_sparse["cat_13"].data.ptr, df_sparse["cat_17"].data.ptr, df_sparse["cat_20"].data.ptr, df_sparse["cat_21"].data.ptr, df_sparse["cat_22"].data.ptr, df_sparse["cat_23"].data.ptr, df_sparse["cat_24"].data.ptr, df_sparse["cat_25"].data.ptr]
        # fill_null_sparse-3 cpu_part_code:
        fill_null_sparse_ptr_list_3_tensor = torch.tensor(fill_null_sparse_ptr_list_3, dtype=torch.int64, device=device)
        # sigrid_hash-4 input_list_code:
        sigrid_hash_input_list_4 = [fill_null_sparse_output_tensor_list_3[0], fill_null_sparse_output_tensor_list_3[1], fill_null_sparse_output_tensor_list_3[2], fill_null_sparse_output_tensor_list_3[3], fill_null_sparse_output_tensor_list_3[4], fill_null_sparse_output_tensor_list_3[5], fill_null_sparse_output_tensor_list_3[6], fill_null_sparse_output_tensor_list_3[7], fill_null_sparse_output_tensor_list_3[8], fill_null_sparse_output_tensor_list_3[9], fill_null_sparse_output_tensor_list_3[10], fill_null_sparse_output_tensor_list_3[11], fill_null_sparse_output_tensor_list_3[12]]
        # sigrid_hash-4 cpu_part_code:
        sigrid_hash_input_ptr_4, sigrid_hash_offset_length_4, sigrid_hash_input_length_4 = cuda_preprocess.sigridhash_cpu_part_base(sigrid_hash_input_list_4)
        # firstx-5 input_list_code:
        firstx_input_list_5 = [sigrid_hash_input_list_4[0], sigrid_hash_input_list_4[1], sigrid_hash_input_list_4[2], sigrid_hash_input_list_4[3], sigrid_hash_input_list_4[4], sigrid_hash_input_list_4[5], sigrid_hash_input_list_4[6], sigrid_hash_input_list_4[7], sigrid_hash_input_list_4[8], sigrid_hash_input_list_4[9], sigrid_hash_input_list_4[10], sigrid_hash_input_list_4[11], sigrid_hash_input_list_4[12]]
        # firstx-5 cpu_part_code:
        firstx_input_ptr_5, firstx_out_ptr_5, firstx_input_length_5, firstx_output_list_5 = cuda_preprocess.firstx_cpu_part(firstx_input_list_5, x_list_5)
        # bucketize-6 input_list_code:
        bucketize_input_list_6 = [fill_null_dense_output_tensor_list_2[0]]
        # bucketize-6 cpu_part_code:
        bucketize_input_ptr_6, bucketize_out_ptr_6, bucketize_input_length_6, bucketize_output_list_6 = cuda_preprocess.bucketize_cpu_part(bucketize_input_list_6, border_list_6)


    # ==================== output_list =====================
    output_list.extend([fill_null_dense_ptr_list_0_tensor, logit_input_ptr_1, logit_input_length_1, fill_null_dense_ptr_list_2_tensor, fill_null_sparse_ptr_list_3_tensor, sigrid_hash_input_ptr_4, sigrid_hash_offset_length_4, sigrid_hash_input_length_4, firstx_input_ptr_5, firstx_out_ptr_5, firstx_input_length_5, firstx_output_list_5, bucketize_input_ptr_6, bucketize_out_ptr_6, bucketize_input_length_6, bucketize_output_list_6])



def thread_cpu_part_gpu_1_next(df_dense, df_sparse, cuda_preprocess, this_rank, device, args, input_list, output_list, dup_times):
    # ==================== input_list =====================
    fill_null_dense_output_tensor_list_7, fill_null_sparse_output_tensor_list_9, x_list_11 = input_list

    cuda_preprocess.init_cuda(this_rank) # set cuda device
    # ==================== input_list + cpu code =====================
    for _ in range(dup_times):
        # fill_null_dense-7 input_list_code:
        fill_null_dense_ptr_list_7 = [df_dense["int_0"].data.ptr, df_dense["int_1"].data.ptr, df_dense["int_2"].data.ptr, df_dense["int_3"].data.ptr, df_dense["int_4"].data.ptr, df_dense["int_5"].data.ptr, df_dense["int_6"].data.ptr, df_dense["int_7"].data.ptr, df_dense["int_8"].data.ptr, df_dense["int_9"].data.ptr, df_dense["int_10"].data.ptr, df_dense["int_11"].data.ptr, df_dense["int_12"].data.ptr]
        # fill_null_dense-7 cpu_part_code:
        fill_null_dense_ptr_list_7_tensor = torch.tensor(fill_null_dense_ptr_list_7, dtype=torch.int64, device=device)
        # logit-8 input_list_code:
        logit_input_list_8 = [fill_null_dense_output_tensor_list_7[0], fill_null_dense_output_tensor_list_7[1], fill_null_dense_output_tensor_list_7[2], fill_null_dense_output_tensor_list_7[3], fill_null_dense_output_tensor_list_7[4], fill_null_dense_output_tensor_list_7[5], fill_null_dense_output_tensor_list_7[6], fill_null_dense_output_tensor_list_7[7], fill_null_dense_output_tensor_list_7[8], fill_null_dense_output_tensor_list_7[9], fill_null_dense_output_tensor_list_7[10], fill_null_dense_output_tensor_list_7[11], fill_null_dense_output_tensor_list_7[12]]
        # logit-8 cpu_part_code:
        logit_input_ptr_8, logit_input_length_8 = cuda_preprocess.logit_cpu_part_base(logit_input_list_8)
        # fill_null_sparse-9 input_list_code:
        fill_null_sparse_ptr_list_9 = [df_sparse["cat_1"].data.ptr, df_sparse["cat_2"].data.ptr, df_sparse["cat_6"].data.ptr, df_sparse["cat_7"].data.ptr, df_sparse["cat_8"].data.ptr, df_sparse["cat_9"].data.ptr, df_sparse["cat_10"].data.ptr, df_sparse["cat_11"].data.ptr, df_sparse["cat_14"].data.ptr, df_sparse["cat_15"].data.ptr, df_sparse["cat_16"].data.ptr, df_sparse["cat_18"].data.ptr, df_sparse["cat_19"].data.ptr]
        # fill_null_sparse-9 cpu_part_code:
        fill_null_sparse_ptr_list_9_tensor = torch.tensor(fill_null_sparse_ptr_list_9, dtype=torch.int64, device=device)
        # sigrid_hash-10 input_list_code:
        sigrid_hash_input_list_10 = [fill_null_sparse_output_tensor_list_9[0], fill_null_sparse_output_tensor_list_9[1], fill_null_sparse_output_tensor_list_9[2], fill_null_sparse_output_tensor_list_9[3], fill_null_sparse_output_tensor_list_9[4], fill_null_sparse_output_tensor_list_9[5], fill_null_sparse_output_tensor_list_9[6], fill_null_sparse_output_tensor_list_9[7], fill_null_sparse_output_tensor_list_9[8], fill_null_sparse_output_tensor_list_9[9], fill_null_sparse_output_tensor_list_9[10], fill_null_sparse_output_tensor_list_9[11], fill_null_sparse_output_tensor_list_9[12]]
        # sigrid_hash-10 cpu_part_code:
        sigrid_hash_input_ptr_10, sigrid_hash_offset_length_10, sigrid_hash_input_length_10 = cuda_preprocess.sigridhash_cpu_part_base(sigrid_hash_input_list_10)
        # firstx-11 input_list_code:
        firstx_input_list_11 = [sigrid_hash_input_list_10[0], sigrid_hash_input_list_10[1], sigrid_hash_input_list_10[2], sigrid_hash_input_list_10[3], sigrid_hash_input_list_10[4], sigrid_hash_input_list_10[5], sigrid_hash_input_list_10[6], sigrid_hash_input_list_10[7], sigrid_hash_input_list_10[8], sigrid_hash_input_list_10[9], sigrid_hash_input_list_10[10], sigrid_hash_input_list_10[11], sigrid_hash_input_list_10[12]]
        # firstx-11 cpu_part_code:
        firstx_input_ptr_11, firstx_out_ptr_11, firstx_input_length_11, firstx_output_list_11 = cuda_preprocess.firstx_cpu_part(firstx_input_list_11, x_list_11)


    # ==================== output_list =====================
    output_list.extend([fill_null_dense_ptr_list_7_tensor, logit_input_ptr_8, logit_input_length_8, fill_null_sparse_ptr_list_9_tensor, sigrid_hash_input_ptr_10, sigrid_hash_offset_length_10, sigrid_hash_input_length_10, firstx_input_ptr_11, firstx_out_ptr_11, firstx_input_length_11, firstx_output_list_11])



def train_and_preprocessing_process(rank, nDev, nProcess, args, queue_list, input_queue_list, if_train) -> None:
    warnings.filterwarnings('ignore')
    random.seed(rank)
    torch.manual_seed(rank)

    if rank < nDev:
        device = torch.device(f"cuda:{rank}")
        pair_rank = rank + nDev
    else:
        device = torch.device(f"cuda:{rank-nDev}")
        this_rank = rank - nDev
        pair_rank = rank - nDev
    
    backend = "gloo"  # synchronization between preprocessing and training process using gloo
    nccl_backend = "nccl"  # synchronization between training process using nccl

    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=nProcess)
    dist.barrier()
    print_once(rank, "finish init global dist")

    train_rank_list = [i for i in range(nDev)]
    preprocess_rank_list = [i+nDev for i in range(nDev)]

    train_group = torch.distributed.new_group(backend='nccl', ranks=train_rank_list)  # group of sub-process for training
    preprocess_group = torch.distributed.new_group(backend='nccl', ranks=preprocess_rank_list)  # group of sub-process for training
    global_group = dist.group.WORLD
    if if_train:
        dist.barrier(group=train_group)
    else:
        dist.barrier(group=preprocess_group)
    print_once(rank, "finish init train and preprocess dist")

    # Updata parameter setting

    if args.preprocessing_plan == 0:
        args.num_embeddings_per_feature = args.num_embeddings_per_feature + [65536]
        args.cat_name = args.cat_name + ["bucketize_int_0"]
        args.nSparse = args.nSparse + 1
    if args.preprocessing_plan == 1:
        args.num_embeddings_per_feature = args.num_embeddings_per_feature + [65536]
        args.cat_name = args.cat_name + ["bucketize_int_0"]
        args.nSparse = args.nSparse + 1
    elif args.preprocessing_plan == 2:
        args.nSparse = 26 * 2
        args.nDense = 13 * 2
        args.cat_name = ["cat_{}".format(i) for i in range(args.nSparse)]
        args.int_name = ["int_" + str(i) for i in range(args.nDense)]
        args.num_embeddings_per_feature = [65536 for _ in range(args.nSparse)]
    elif args.preprocessing_plan == 3:
        args.nSparse = 26 * 4
        args.nDense = 13 * 4
        args.cat_name = ["cat_{}".format(i) for i in range(args.nSparse)]
        args.int_name = ["int_" + str(i) for i in range(args.nDense)]
        args.num_embeddings_per_feature = [65536 for _ in range(args.nSparse)]

    if if_train:
        torch.cuda.set_device(rank)

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=args.embedding_dim,
                num_embeddings = args.num_embeddings_per_feature[feature_idx],
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(args.cat_name)
        ]

        embedding_tables = EmbeddingBagCollection(
            tables=eb_configs,
            device=torch.device("meta"), # "meta" model will not allocate memory until sharding
        )

        sharding_constraints = {
            f"t_{feature_name}": ParameterConstraints(
            sharding_types=[ShardingType.TABLE_WISE.value],  # TABLE_WISE, ROW_WISE, COLUMN_WISE, DATA_PARALLEL
            ) for feature_idx, feature_name in enumerate(args.cat_name)
        }

        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=nDev,
                world_size=nDev,
                compute_device=device.type,
            ),
            batch_size=args.batch_size,
            storage_reservation=HeuristicalStorageReservation(percentage=0.01),
            constraints=sharding_constraints,
        )

        plan = planner.collective_plan(
            module=embedding_tables, sharders=get_default_sharders(), pg=train_group
        )

        # =================== Make sure sharding is correct ===================
        nTable = len(args.cat_name)
        table_mapping = []
        if args.preprocessing_plan in [0, 1] and args.nDev == 2:
            table_mapping = [0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0] # default sharding for preprocessing_plan=0
        elif args.preprocessing_plan in [0, 1] and args.nDev == 4:
            table_mapping = [1,1,3,2, 2,2,3,1, 1,3,1,3, 0,0,1,3, 1,2,3,0, 0,2,2,0, 0,2,0]
        elif args.preprocessing_plan in [0, 1] and args.nDev == 8:
            table_mapping = [1, 1, 3, 6, 2, 2, 7, 1, 5, 3, 5, 7, 0, 0, 5, 3, 1, 2, 7, 0, 4, 2, 6, 4, 4, 6, 0]
        elif args.preprocessing_plan == 2 and args.nDev == 2:
            table_mapping = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        elif args.preprocessing_plan == 2 and args.nDev == 4:
            table_mapping = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
        elif args.preprocessing_plan == 2 and args.nDev == 8:
            table_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3]
        elif args.preprocessing_plan == 3 and args.nDev == 2:
            table_mapping = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        elif args.preprocessing_plan == 3 and args.nDev == 4:
            table_mapping = [1, 2, 0, 3, 3, 3, 0, 2, 2, 3, 1, 3, 1, 1, 2, 0, 2, 3, 0, 0, 0, 2, 2, 1, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        elif args.preprocessing_plan == 3 and args.nDev == 8:
            table_mapping = [1, 2, 0, 3, 3, 3, 0, 2, 2, 3, 1, 3, 1, 1, 2, 0, 2, 3, 0, 0, 0, 2, 2, 1, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
        else:
            raise ValueError("args.preprocessing_plan or args.nDev is not supported")

        if nTable != len(table_mapping):
            raise ValueError("nTable != len(table_mapping)")
        table_names = plan.plan[''].keys()

        for idx, t_name in enumerate(table_names):
            plan.plan[''][t_name].ranks = [table_mapping[idx]]
            plan.plan[''][t_name].sharding_spec.shards[0].placement._device = torch.device("cuda:{}".format(table_mapping[idx]))
            plan.plan[''][t_name].sharding_spec.shards[0].placement._rank = table_mapping[idx]

        # if rank == 0:
            # print(plan)  
        #====================================================

        sharded_emts = ShardedEmbeddingTable(
            embedding_tables=embedding_tables, 
            plan=plan, 
            device=device, 
            group=train_group, 
            dim=args.embedding_dim,
            batch_size=args.batch_size, 
            table_names=args.cat_name,
            input_queue=queue_list[rank],
        )

        mlp_layers = DLRM_MLP(
            embedding_dim=args.embedding_dim,
            num_sparse_features=args.nSparse,
            dense_in_features=args.nDense,
            dense_arch_layer_sizes=args.dense_arch_layer_sizes,
            over_arch_layer_sizes=args.over_arch_layer_sizes,
            dense_device=device,
        )

        sparse_optimizer = KeyedOptimizerWrapper(
            dict(sharded_emts.named_parameters()),
            lambda params: torch.optim.SGD(params, lr=args.learning_rate),
        )
        dense_optimizer = optim.SGD(mlp_layers.parameters(), lr=args.learning_rate)

        loss_fn = torch.nn.BCEWithLogitsLoss()

        

        train_dataloader = get_dataloader(args, backend, "train")
        in_mem_dataloader = InMemoryDataLoader(train_dataloader, rank, nDev, 16)

        batch = in_mem_dataloader.next()
        dist_input = sharded_emts.input_comm(batch.sparse_features) # dist_input[0] is JaggedTensor
        # print("rank", rank, "dist_input[0]._values.shape: ", dist_input[0]._values.shape)

        dist.barrier(group=train_group)
        print_once(rank, "finish model initalization")

    else:
        table_length_dic = {}
        for idx, table_name in enumerate(args.cat_name):
            table_length_dic[table_name] = args.num_embeddings_per_feature[idx]

        cuda_preprocess = load(name="gpu_operators", sources=[
        "/workspace/RAP/cuda_operators/cuda_wrap.cpp", 
        "/workspace/RAP/cuda_operators/gpu_operators.cu", 
        ], verbose=False)

        cuda_preprocess.init_cuda(this_rank)

        dup_times = 1
        # if args.preprocessing_plan == 2:
        #     dup_times = 2
        # elif args.preprocessing_plan == 3:
        #     dup_times = 4

        label_name = "label"
        sparse_name = [f"cat_{i}" for i in range(26)]
        dense_name = args.int_name

        data_dir = "/workspace/RAP/RAP_end_to_end/splitted_input/"
        dense_file_name = data_dir + "GPU_{}_dense_{}.parquet".format(this_rank, args.preprocessing_plan)
        sparse_file_name = data_dir + "GPU_{}_sparse_{}.parquet".format(this_rank, args.preprocessing_plan)

        # ===============================  Pointer Prepare  ====================================================
        df_sparse = cudf.read_parquet(sparse_file_name)
        df_dense = cudf.read_parquet(dense_file_name)
        if this_rank == 0:
            # ================= CPU part code =================
            # fill_null_dense-0 input_list_code:
            fill_null_dense_ptr_list_0 = [df_dense["int_0"].data.ptr, df_dense["int_1"].data.ptr, df_dense["int_2"].data.ptr, df_dense["int_3"].data.ptr, df_dense["int_4"].data.ptr, df_dense["int_5"].data.ptr, df_dense["int_6"].data.ptr, df_dense["int_7"].data.ptr, df_dense["int_8"].data.ptr, df_dense["int_9"].data.ptr, df_dense["int_10"].data.ptr, df_dense["int_11"].data.ptr, df_dense["int_12"].data.ptr]
            # fill_null_dense-0 data_prepare_code:
            fill_null_dense_output_tensor_list_0 = [torch.zeros((args.batch_size, 1), dtype=torch.float, device=device) for _ in range(len(fill_null_dense_ptr_list_0))]
            fill_null_dense_output_tensor_list_0_ptr = cuda_preprocess.copy_tensor_list_to_GPU_tensor(fill_null_dense_output_tensor_list_0)
            # fill_null_dense-0 cpu_part_code:
            fill_null_dense_ptr_list_0_tensor = torch.tensor(fill_null_dense_ptr_list_0, dtype=torch.int64, device=device)
            # logit-1 input_list_code:
            logit_input_list_1 = [fill_null_dense_output_tensor_list_0[0], fill_null_dense_output_tensor_list_0[1], fill_null_dense_output_tensor_list_0[2], fill_null_dense_output_tensor_list_0[3], fill_null_dense_output_tensor_list_0[4], fill_null_dense_output_tensor_list_0[5], fill_null_dense_output_tensor_list_0[6], fill_null_dense_output_tensor_list_0[7], fill_null_dense_output_tensor_list_0[8], fill_null_dense_output_tensor_list_0[9], fill_null_dense_output_tensor_list_0[10], fill_null_dense_output_tensor_list_0[11], fill_null_dense_output_tensor_list_0[12]]
            # logit-1 data_prepare_code:
            logit_eps_list_1 = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
            logit_eps_list_ptr_1 = cuda_preprocess.logit_ptr_prepare(logit_input_list_1, logit_eps_list_1)
            # logit-1 cpu_part_code:
            logit_input_ptr_1, logit_input_length_1 = cuda_preprocess.logit_cpu_part_base(logit_input_list_1)
            # fill_null_dense-2 input_list_code:
            fill_null_dense_ptr_list_2 = [df_sparse["int_0"].data.ptr]
            # fill_null_dense-2 data_prepare_code:
            fill_null_dense_output_tensor_list_2 = [torch.zeros((args.batch_size * args.nDev, 1), dtype=torch.float, device=device) for _ in range(len(fill_null_dense_ptr_list_2))]
            fill_null_dense_output_tensor_list_2_ptr = cuda_preprocess.copy_tensor_list_to_GPU_tensor(fill_null_dense_output_tensor_list_2)
            # fill_null_dense-2 cpu_part_code:
            fill_null_dense_ptr_list_2_tensor = torch.tensor(fill_null_dense_ptr_list_2, dtype=torch.int64, device=device)
            # fill_null_sparse-3 input_list_code:
            fill_null_sparse_ptr_list_3 = [df_sparse["cat_0"].data.ptr, df_sparse["cat_3"].data.ptr, df_sparse["cat_4"].data.ptr, df_sparse["cat_5"].data.ptr, df_sparse["cat_12"].data.ptr, df_sparse["cat_13"].data.ptr, df_sparse["cat_17"].data.ptr, df_sparse["cat_20"].data.ptr, df_sparse["cat_21"].data.ptr, df_sparse["cat_22"].data.ptr, df_sparse["cat_23"].data.ptr, df_sparse["cat_24"].data.ptr, df_sparse["cat_25"].data.ptr]
            # fill_null_sparse-3 data_prepare_code:
            fill_null_sparse_output_tensor_list_3 = [torch.zeros((args.batch_size * args.nDev, 1), dtype=torch.int64, device=device) for _ in range(len(fill_null_sparse_ptr_list_3))]
            fill_null_sparse_output_tensor_list_3_ptr = cuda_preprocess.copy_tensor_list_to_GPU_tensor(fill_null_sparse_output_tensor_list_3)
            # fill_null_sparse-3 cpu_part_code:
            fill_null_sparse_ptr_list_3_tensor = torch.tensor(fill_null_sparse_ptr_list_3, dtype=torch.int64, device=device)
            # sigrid_hash-4 input_list_code:
            sigrid_hash_input_list_4 = [fill_null_sparse_output_tensor_list_3[0], fill_null_sparse_output_tensor_list_3[1], fill_null_sparse_output_tensor_list_3[2], fill_null_sparse_output_tensor_list_3[3], fill_null_sparse_output_tensor_list_3[4], fill_null_sparse_output_tensor_list_3[5], fill_null_sparse_output_tensor_list_3[6], fill_null_sparse_output_tensor_list_3[7], fill_null_sparse_output_tensor_list_3[8], fill_null_sparse_output_tensor_list_3[9], fill_null_sparse_output_tensor_list_3[10], fill_null_sparse_output_tensor_list_3[11], fill_null_sparse_output_tensor_list_3[12]]
            # sigrid_hash-4 data_prepare_code:
            sigrid_hash_length_list_4 = [1460, 2202608, 305, 24, 3194, 27, 5652, 7046547, 18, 15, 286181, 105, 142572]
            gpu_table_list_4, gpu_multiplier_list_4, gpu_shift_list_4 = cuda_preprocess.sigridhash_ptr_prepare(sigrid_hash_length_list_4)
            # sigrid_hash-4 cpu_part_code:
            sigrid_hash_input_ptr_4, sigrid_hash_offset_length_4, sigrid_hash_input_length_4 = cuda_preprocess.sigridhash_cpu_part_base(sigrid_hash_input_list_4)
            # firstx-5 input_list_code:
            firstx_input_list_5 = [sigrid_hash_input_list_4[0], sigrid_hash_input_list_4[1], sigrid_hash_input_list_4[2], sigrid_hash_input_list_4[3], sigrid_hash_input_list_4[4], sigrid_hash_input_list_4[5], sigrid_hash_input_list_4[6], sigrid_hash_input_list_4[7], sigrid_hash_input_list_4[8], sigrid_hash_input_list_4[9], sigrid_hash_input_list_4[10], sigrid_hash_input_list_4[11], sigrid_hash_input_list_4[12]]
            # firstx-5 data_prepare_code:
            x_list_5 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            x_list_ptr_5, firstx_width_list_ptr_5 = cuda_preprocess.firstx_ptr_prepare(firstx_input_list_5, x_list_5)
            # firstx-5 cpu_part_code:
            firstx_input_ptr_5, firstx_out_ptr_5, firstx_input_length_5, firstx_output_list_5 = cuda_preprocess.firstx_cpu_part(firstx_input_list_5, x_list_5)
            # bucketize-6 input_list_code:
            bucketize_input_list_6 = [fill_null_dense_output_tensor_list_2[0]]
            # bucketize-6 data_prepare_code:
            border_list_6 = [torch.tensor([0.5, 1.0, 1.5], dtype=torch.float32, device=device)]
            bucketize_border_list_ptr_6, bucketize_length_list_ptr_6 = cuda_preprocess.bucketize_ptr_prepare(bucketize_input_list_6, border_list_6)
            # bucketize-6 cpu_part_code:
            bucketize_input_ptr_6, bucketize_out_ptr_6, bucketize_input_length_6, bucketize_output_list_6 = cuda_preprocess.bucketize_cpu_part(bucketize_input_list_6, border_list_6)
            # ==========================================================================
            # =========================== First Iter finished ==========================
            # ==========================================================================


            # ================= prepare input list for thread function ================= 
            input_list = [fill_null_dense_output_tensor_list_0, fill_null_dense_output_tensor_list_2, fill_null_sparse_output_tensor_list_3, x_list_5, border_list_6]

            
        elif this_rank == 1:
            # ================= CPU part code =================
            # fill_null_dense-7 input_list_code:
            fill_null_dense_ptr_list_7 = [df_dense["int_0"].data.ptr, df_dense["int_1"].data.ptr, df_dense["int_2"].data.ptr, df_dense["int_3"].data.ptr, df_dense["int_4"].data.ptr, df_dense["int_5"].data.ptr, df_dense["int_6"].data.ptr, df_dense["int_7"].data.ptr, df_dense["int_8"].data.ptr, df_dense["int_9"].data.ptr, df_dense["int_10"].data.ptr, df_dense["int_11"].data.ptr, df_dense["int_12"].data.ptr]
            # fill_null_dense-7 data_prepare_code:
            fill_null_dense_output_tensor_list_7 = [torch.zeros((args.batch_size, 1), dtype=torch.float, device=device) for _ in range(len(fill_null_dense_ptr_list_7))]
            fill_null_dense_output_tensor_list_7_ptr = cuda_preprocess.copy_tensor_list_to_GPU_tensor(fill_null_dense_output_tensor_list_7)
            # fill_null_dense-7 cpu_part_code:
            fill_null_dense_ptr_list_7_tensor = torch.tensor(fill_null_dense_ptr_list_7, dtype=torch.int64, device=device)
            # logit-8 input_list_code:
            logit_input_list_8 = [fill_null_dense_output_tensor_list_7[0], fill_null_dense_output_tensor_list_7[1], fill_null_dense_output_tensor_list_7[2], fill_null_dense_output_tensor_list_7[3], fill_null_dense_output_tensor_list_7[4], fill_null_dense_output_tensor_list_7[5], fill_null_dense_output_tensor_list_7[6], fill_null_dense_output_tensor_list_7[7], fill_null_dense_output_tensor_list_7[8], fill_null_dense_output_tensor_list_7[9], fill_null_dense_output_tensor_list_7[10], fill_null_dense_output_tensor_list_7[11], fill_null_dense_output_tensor_list_7[12]]
            # logit-8 data_prepare_code:
            logit_eps_list_8 = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
            logit_eps_list_ptr_8 = cuda_preprocess.logit_ptr_prepare(logit_input_list_8, logit_eps_list_8)
            # logit-8 cpu_part_code:
            logit_input_ptr_8, logit_input_length_8 = cuda_preprocess.logit_cpu_part_base(logit_input_list_8)
            # fill_null_sparse-9 input_list_code:
            fill_null_sparse_ptr_list_9 = [df_sparse["cat_1"].data.ptr, df_sparse["cat_2"].data.ptr, df_sparse["cat_6"].data.ptr, df_sparse["cat_7"].data.ptr, df_sparse["cat_8"].data.ptr, df_sparse["cat_9"].data.ptr, df_sparse["cat_10"].data.ptr, df_sparse["cat_11"].data.ptr, df_sparse["cat_14"].data.ptr, df_sparse["cat_15"].data.ptr, df_sparse["cat_16"].data.ptr, df_sparse["cat_18"].data.ptr, df_sparse["cat_19"].data.ptr]
            # fill_null_sparse-9 data_prepare_code:
            fill_null_sparse_output_tensor_list_9 = [torch.zeros((args.batch_size * args.nDev, 1), dtype=torch.int64, device=device) for _ in range(len(fill_null_sparse_ptr_list_9))]
            fill_null_sparse_output_tensor_list_9_ptr = cuda_preprocess.copy_tensor_list_to_GPU_tensor(fill_null_sparse_output_tensor_list_9)
            # fill_null_sparse-9 cpu_part_code:
            fill_null_sparse_ptr_list_9_tensor = torch.tensor(fill_null_sparse_ptr_list_9, dtype=torch.int64, device=device)
            # sigrid_hash-10 input_list_code:
            sigrid_hash_input_list_10 = [fill_null_sparse_output_tensor_list_9[0], fill_null_sparse_output_tensor_list_9[1], fill_null_sparse_output_tensor_list_9[2], fill_null_sparse_output_tensor_list_9[3], fill_null_sparse_output_tensor_list_9[4], fill_null_sparse_output_tensor_list_9[5], fill_null_sparse_output_tensor_list_9[6], fill_null_sparse_output_tensor_list_9[7], fill_null_sparse_output_tensor_list_9[8], fill_null_sparse_output_tensor_list_9[9], fill_null_sparse_output_tensor_list_9[10], fill_null_sparse_output_tensor_list_9[11], fill_null_sparse_output_tensor_list_9[12]]
            # sigrid_hash-10 data_prepare_code:
            sigrid_hash_length_list_10 = [583, 10131227, 12517, 633, 3, 93145, 5683, 8351593, 14992, 5461306, 10, 2173, 4]
            gpu_table_list_10, gpu_multiplier_list_10, gpu_shift_list_10 = cuda_preprocess.sigridhash_ptr_prepare(sigrid_hash_length_list_10)
            # sigrid_hash-10 cpu_part_code:
            sigrid_hash_input_ptr_10, sigrid_hash_offset_length_10, sigrid_hash_input_length_10 = cuda_preprocess.sigridhash_cpu_part_base(sigrid_hash_input_list_10)
            # firstx-11 input_list_code:
            firstx_input_list_11 = [sigrid_hash_input_list_10[0], sigrid_hash_input_list_10[1], sigrid_hash_input_list_10[2], sigrid_hash_input_list_10[3], sigrid_hash_input_list_10[4], sigrid_hash_input_list_10[5], sigrid_hash_input_list_10[6], sigrid_hash_input_list_10[7], sigrid_hash_input_list_10[8], sigrid_hash_input_list_10[9], sigrid_hash_input_list_10[10], sigrid_hash_input_list_10[11], sigrid_hash_input_list_10[12]]
            # firstx-11 data_prepare_code:
            x_list_11 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            x_list_ptr_11, firstx_width_list_ptr_11 = cuda_preprocess.firstx_ptr_prepare(firstx_input_list_11, x_list_11)
            # firstx-11 cpu_part_code:
            firstx_input_ptr_11, firstx_out_ptr_11, firstx_input_length_11, firstx_output_list_11 = cuda_preprocess.firstx_cpu_part(firstx_input_list_11, x_list_11)
            # ==========================================================================
            # =========================== First Iter finished ==========================
            # ==========================================================================


            # ================= prepare input list for thread function ================= 
            input_list = [fill_null_dense_output_tensor_list_7, fill_null_sparse_output_tensor_list_9, x_list_11]

            
        dist.barrier(group=preprocess_group)
        print_once(this_rank, "finish cudf loading initalization")

    dist.barrier()
    print_once(rank, "Finish all initalization, start training")

    
    n_train = 1024
    n_warmup = 1024
    n_loop = n_train + n_warmup
    loop_range = range(n_loop)
    if rank == 0:
        loop_range = tqdm(loop_range)
    if not if_train:
        dense_date_list = [df_dense] + [None for _ in range(n_loop)]
        sparse_date_list = [df_sparse] + [None for _ in range(n_loop)]
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in loop_range:
        if i == n_warmup and rank == 0:
            start_event.record()
        if if_train:
            if i > 0: # skip the first iter
                dense_input, sparse_input, label_tensor = input_queue_list[rank].get()
                # dist_input.features[0]._values = sparse_input.clone()

                training_iter_without_input_dist(queue_list, sparse_optimizer, dense_optimizer, loss_fn, dist_input, dense_input, label_tensor, sharded_emts, mlp_layers, rank, train_group)

                # del dense_input, sparse_input, label_tensor
        else:
            dense_date_list[1] = cudf.read_parquet(dense_file_name)
            sparse_date_list[1] = cudf.read_parquet(sparse_file_name)
            output_list = []
            if this_rank == 0:
                # ================= CPU part code =================
                thread = threading.Thread(target=thread_cpu_part_gpu_0_next, args=(dense_date_list[1], sparse_date_list[1], cuda_preprocess, this_rank, device, args, input_list, output_list, dup_times))
                thread.start()
                # ================= GPU part code =================
                if i > 0:
                    queue_wait(queue_list[this_rank])
                for _ in range(dup_times):
                    # fill_null_dense-0 gpu_code:
                    cuda_preprocess.fill_null_float_list_gpu_part_tensor(fill_null_dense_ptr_list_0_tensor, fill_null_dense_output_tensor_list_0_ptr, args.batch_size)
                    # logit-1 gpu_code:
                    cuda_preprocess.logit_gpu_part(logit_input_ptr_1, logit_eps_list_ptr_1, logit_input_length_1, args.batch_size)
                    # fill_null_dense-2 gpu_code:
                    cuda_preprocess.fill_null_float_list_gpu_part_tensor(fill_null_dense_ptr_list_2_tensor, fill_null_dense_output_tensor_list_2_ptr, args.batch_size * args.nDev)
                    # fill_null_sparse-3 gpu_code:
                    cuda_preprocess.fill_null_int64_list_gpu_part_tensor(fill_null_sparse_ptr_list_3_tensor, fill_null_sparse_output_tensor_list_3_ptr, args.batch_size * args.nDev)
                    # sigrid_hash-4 gpu_code:
                    cuda_preprocess.sigridhash_gpu_part(sigrid_hash_input_ptr_4, sigrid_hash_offset_length_4, 0, gpu_table_list_4, gpu_multiplier_list_4, gpu_shift_list_4, sigrid_hash_input_length_4, args.batch_size * args.nDev)
                    # firstx-5 gpu_code:
                    cuda_preprocess.firstx_gpu_part(firstx_input_ptr_5, firstx_out_ptr_5, x_list_ptr_5, firstx_width_list_ptr_5, firstx_input_length_5, args.batch_size * args.nDev, 0)
                    # bucketize-6 gpu_code:
                    cuda_preprocess.bucketize_gpu_part(bucketize_input_ptr_6, bucketize_out_ptr_6, bucketize_border_list_ptr_6, bucketize_length_list_ptr_6, bucketize_input_length_6, args.batch_size * args.nDev)

                # ================= output tensor prepare code =================
                dense_input_list = [logit_input_list_1[0], logit_input_list_1[1], logit_input_list_1[2], logit_input_list_1[3], logit_input_list_1[4], logit_input_list_1[5], logit_input_list_1[6], logit_input_list_1[7], logit_input_list_1[8], logit_input_list_1[9], logit_input_list_1[10], logit_input_list_1[11], logit_input_list_1[12]]

                sparse_input_list = [firstx_output_list_5[0], firstx_output_list_5[1], firstx_output_list_5[2], firstx_output_list_5[3], firstx_output_list_5[4], firstx_output_list_5[5], firstx_output_list_5[6], firstx_output_list_5[7], firstx_output_list_5[8], firstx_output_list_5[9], firstx_output_list_5[10], firstx_output_list_5[11], firstx_output_list_5[12], bucketize_output_list_6[0]]


                if i < n_loop - 1:
                    label_tensor = cuda_preprocess.fill_null_float(df_dense[label_name].data.ptr, args.batch_size).squeeze(1)
                    sparse_input = torch.cat(sparse_input_list, dim=0).squeeze(1)
                    dense_input = torch.cat(dense_input_list, dim=1)
                    input_queue_list[this_rank].put((dense_input, sparse_input, label_tensor)) 

                # ================= Get ptrs for the next batch ================= 
                thread.join()
                if len(output_list) == 16:
                    fill_null_dense_ptr_list_0_tensor, logit_input_ptr_1, logit_input_length_1, fill_null_dense_ptr_list_2_tensor, fill_null_sparse_ptr_list_3_tensor, sigrid_hash_input_ptr_4, sigrid_hash_offset_length_4, sigrid_hash_input_length_4, firstx_input_ptr_5, firstx_out_ptr_5, firstx_input_length_5, firstx_output_list_5, bucketize_input_ptr_6, bucketize_out_ptr_6, bucketize_input_length_6, bucketize_output_list_6 = output_list



            elif this_rank == 1:
                # ================= CPU part code =================
                thread = threading.Thread(target=thread_cpu_part_gpu_1_next, args=(dense_date_list[1], sparse_date_list[1], cuda_preprocess, this_rank, device, args, input_list, output_list, dup_times))
                thread.start()
                # ================= GPU part code =================
                if i > 0:
                    queue_wait(queue_list[this_rank])
                for _ in range(dup_times):
                    # fill_null_dense-7 gpu_code:
                    cuda_preprocess.fill_null_float_list_gpu_part_tensor(fill_null_dense_ptr_list_7_tensor, fill_null_dense_output_tensor_list_7_ptr, args.batch_size)
                    # logit-8 gpu_code:
                    cuda_preprocess.logit_gpu_part(logit_input_ptr_8, logit_eps_list_ptr_8, logit_input_length_8, args.batch_size)
                    # fill_null_sparse-9 gpu_code:
                    cuda_preprocess.fill_null_int64_list_gpu_part_tensor(fill_null_sparse_ptr_list_9_tensor, fill_null_sparse_output_tensor_list_9_ptr, args.batch_size * args.nDev)
                    # sigrid_hash-10 gpu_code:
                    cuda_preprocess.sigridhash_gpu_part(sigrid_hash_input_ptr_10, sigrid_hash_offset_length_10, 0, gpu_table_list_10, gpu_multiplier_list_10, gpu_shift_list_10, sigrid_hash_input_length_10, args.batch_size * args.nDev)
                    # firstx-11 gpu_code:
                    cuda_preprocess.firstx_gpu_part(firstx_input_ptr_11, firstx_out_ptr_11, x_list_ptr_11, firstx_width_list_ptr_11, firstx_input_length_11, args.batch_size * args.nDev, 0)

                # ================= output tensor prepare code =================
                dense_input_list = [logit_input_list_8[0], logit_input_list_8[1], logit_input_list_8[2], logit_input_list_8[3], logit_input_list_8[4], logit_input_list_8[5], logit_input_list_8[6], logit_input_list_8[7], logit_input_list_8[8], logit_input_list_8[9], logit_input_list_8[10], logit_input_list_8[11], logit_input_list_8[12]]

                sparse_input_list = [firstx_output_list_11[0], firstx_output_list_11[1], firstx_output_list_11[2], firstx_output_list_11[3], firstx_output_list_11[4], firstx_output_list_11[5], firstx_output_list_11[6], firstx_output_list_11[7], firstx_output_list_11[8], firstx_output_list_11[9], firstx_output_list_11[10], firstx_output_list_11[11], firstx_output_list_11[12]]


                if i < n_loop - 1:
                    label_tensor = cuda_preprocess.fill_null_float(df_dense[label_name].data.ptr, args.batch_size).squeeze(1)
                    sparse_input = torch.cat(sparse_input_list, dim=0).squeeze(1)
                    dense_input = torch.cat(dense_input_list, dim=1)
                    input_queue_list[this_rank].put((dense_input, sparse_input, label_tensor)) 

                # ================= Get ptrs for the next batch ================= 
                thread.join()
                if len(output_list) == 11:
                    fill_null_dense_ptr_list_7_tensor, logit_input_ptr_8, logit_input_length_8, fill_null_sparse_ptr_list_9_tensor, sigrid_hash_input_ptr_10, sigrid_hash_offset_length_10, sigrid_hash_input_length_10, firstx_input_ptr_11, firstx_out_ptr_11, firstx_input_length_11, firstx_output_list_11 = output_list


            # free the cudf data frame
            del dense_date_list[0]
            del sparse_date_list[0]

    if rank == 0:
        end_event.record()
        torch.cuda.synchronize()
        avg_latency = start_event.elapsed_time(end_event) / n_train
        through_put = 1 / (avg_latency / 1000)
        print("avg latency:{:.3f} ms, throughput_per_GPU: {:.3f} iter/s, total_throughput:{:.3f} iter/s".format(avg_latency, through_put, through_put * nDev))
        file_name = "result/result_GPU-{}_Plan-{}_Batch-{}_{}.log".format(nDev, args.preprocessing_plan, args.batch_size, os.getpid())
        with open(file_name, 'w') as file:
            file.write("rank:{}, avg_latency:{:.3f} ms/iter, throughput:{:.3f} iters/s, total_throughput:{:.3f}".format(rank, avg_latency, through_put, through_put*nDev))
    dist.barrier()

if __name__ == "__main__":
    random.seed(123)
    torch.manual_seed(123)

    args = get_args()
    nDev = args.nDev

    processes = []
    mp.set_start_method("spawn")

    queue_list = []
    for i in range(nDev):
        queue_list.append(mp.SimpleQueue())

    input_queue_list = []
    for i in range(nDev):
        input_queue_list.append(mp.Queue())

    nProcess = nDev * 2

    for i in range(nProcess):
        if i < nDev:
            if_train = True
        else:
            if_train = False

        p = mp.Process(target=train_and_preprocessing_process, args=(i, nDev, nProcess, args, queue_list, input_queue_list, if_train))
        processes.append(p)

    for p in processes:
        p.start()

    for p in processes:
        p.join()