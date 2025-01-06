import argparse
import os
import pickle
import shutil
from contextlib import nullcontext

import psutil
import ray
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

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(16, 16)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 8)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))
    
def setup(rank, world_size, use_ray=False):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355 "

    # initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    if not use_ray:
        torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def get_reshard_after_forward(sharding_strategy):
    if sharding_strategy == "FULL_SHARD":
        return True
    elif sharding_strategy == "SHARD_GRAD_OP":
        return False
    else:
        raise ValueError(f"Invalid sharding strategy for FSDP v2: {sharding_strategy}")
    

def run_fsdp_example(rank, world_size, args):
    setup(rank, world_size)
    

    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )
    process = psutil.Process()
    peak_mem = 0
    if args.profile:
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(args.checkpoint_dir, "trace")
            ),
        )
    else:
        prof = nullcontext()
    with prof:
        record_function = (
            torch.autograd.profiler.record_function("creation")
            if args.profile
            else nullcontext()
        )
        with record_function:
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

        cuda_info = torch.cuda.memory_stats("cuda")
        max_active = cuda_info["active_bytes.all.peak"]
        max_reserved = cuda_info["reserved_bytes.all.peak"]
        if rank == 0:
            print("Before step memory allocated", (max_active, max_reserved))
        record_function = (
            torch.autograd.profiler.record_function("run")
            if args.profile
            else nullcontext()
        )

        with record_function:
            model(
                input_ids=torch.randint(0, 100, (1, 16), device="cuda")
            ).logits.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
            # clear activation state
            torch.cuda.empty_cache()
        max_active = cuda_info["active_bytes.all.peak"]
        max_reserved = cuda_info["reserved_bytes.all.peak"]
        if rank == 0:
            print("After step memory allocated", (max_active, max_reserved))
    snapshot = torch.cuda.memory._snapshot()
    with open(f"{args.checkpoint_dir}/snapshot_{rank}.pkl", "wb") as f:
        pickle.dump(snapshot, f)
    
    # synchronize 
    torch.distributed.barrier()

    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixed_precision", action="store_true", help="Use mixed precision"
    )
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16")
    parser.add_argument(
        "--world_size", type=int, default=torch.cuda.device_count(), help="World size"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Profile with torch profiler"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=CHECKPOINT_DIR,
        help="Checkpoint directory",
    )
    parser.add_argument("--model_id", type=str, default=MODEL_ID, help="Model id")
    parser.add_argument(
        "--sharding_strategy",
        type=str,
        default="FULL_SHARD",
        help=f"Sharding strategy. Options: {[v for v in ShardingStrategy.__members__.values()]}",
    )
    parser.add_argument("--use_ray", action="store_true", help="Use ray")
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Running fsdp example on {world_size} devices.")
    if os.path.exists(args.checkpoint_dir):
        print(
            f"Checkpoint directory {args.checkpoint_dir} already exists. Removing current contents."
        )
        shutil.rmtree(args.checkpoint_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.use_ray:
        ray.init()
        func = ray.remote(num_gpus=1)(run_fsdp_example)
        ray.get([func.remote(i, world_size, args) for i in range(world_size)])
    else:
        mp.spawn(
            run_fsdp_example,
            args=(world_size, args),
            nprocs=world_size,
            join=True,
        )