import torch 

from .activation_checkpointing_functions import apply_fsdp_checkpointing
from .mixed_precision import bfSixteen, fpSixteen
from .wrapping import get_transformer_wrapper


def get_policies_v1(cfg, model, rank):
    """establish current policies for mixed precision and fsdp wrapping for FSDP v1"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.mixed_precision:
        # bfloat_available = bfloat_support()
        if not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled. ")

    # hardcoded for now
    wrapping_policy = get_transformer_wrapper(model)

    return mixed_precision_policy, wrapping_policy

def get_policies_v2(cfg, model, rank):
    """establish current policies for mixed precision and fsdp wrapping for FSDP v2"""
    from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy

    mixed_precision_policy = None
    # auto wrap not supported in v2
    wrapping_policy = None
     # mixed precision -----
    if cfg.mixed_precision:
        # bfloat_available = bfloat_support()
        if not cfg.use_fp16:
            mixed_precision_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, output_dtype=torch.bfloat16)
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            mixed_precision_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float16, output_dtype=torch.float16)
            if rank == 0:
                print("FP16 enabled. ")
        
    return mixed_precision_policy, wrapping_policy

__all__ = [
    "apply_fsdp_checkpointing",
    "bfSixteen",
    "fpSixteen",
    "get_transformer_wrapper",
    "get_policies_v1",
]