import argparse
import os
import pickle
import shutil
from contextlib import nullcontext

import psutil
import ray
from ray import train
from ray.train import TorchTrainer
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from policies import get_policies_v2
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp import ShardingStrategy
from transformers import AutoModelForCausalLM
from policies.wrapping import get_transformer_cls_to_wrap

MODEL_ID = "BEE-spoke-data/smol_llama-101M-GQA"

CHECKPOINT_DIR = "/mnt/local_nvme/sumanth/test_out/checkpoint_v2"
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT = 100000

def get_reshard_after_forward(sharding_strategy):
    if sharding_strategy == "FULL_SHARD":
        return True
    elif sharding_strategy == "SHARD_GRAD_OP":
        return False
    else:
        raise ValueError(f"Invalid sharding strategy for FSDP v2: {sharding_strategy}")
    

def run_fsdp_example(config):
    args = config["args"]
    world_size = ray.train.get_context().get_world_size()
    rank = ray.train.get_context().get_global_rank()

    reshard_after_forward = get_reshard_after_forward(args.sharding_strategy)
    # create a model and move it to GPU with id rank
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, low_cpu_mem_usage=True
    )
    mixed_precision_policy, _ = get_policies_v2(args, model, rank)
    if rank == 0:
        print(mixed_precision_policy)
    fsdp_kwargs =  dict(mp_policy=mixed_precision_policy, reshard_after_forward=reshard_after_forward)
    #fsdpv2 doesn't have auto-wrap policy, we go inside ourselves 
    cls_to_wrap = tuple(get_transformer_cls_to_wrap(model))
    for module in model.modules():
        if isinstance(module, cls_to_wrap):
            # fsdpv2 is in-place
            fully_shard(module, **fsdp_kwargs)
    # fsdpv2 is in-place
    fully_shard(
        model,
        **fsdp_kwargs,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)


    model(
        input_ids=torch.randint(0, 100, (1, 16), device="cuda")
    ).logits.mean().backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision"
    )
    parser.add_argument(
        "--num_devices", type=int, default=torch.cuda.device_count(), help="World size"
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Model id")
    parser.add_argument(
        "--sharding_strategy",
        type=str,
        default="FULL_SHARD",
        help=f"Sharding strategy. Options: {[v for v in ShardingStrategy.__members__.values()]}",
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Running fsdp example on {world_size} devices.")
    trainer =  TorchTrainer(
        train_loop_per_worker=run_fsdp_example,
        train_loop_config={"args": args},
        scaling_config=train.ScalingConfig(
            num_workers=args.num_devices,
            use_gpu=True,
            resources_per_worker={"GPU": 1},
        ),
    )
    trainer.fit()