from .activation_checkpointing_functions import apply_fsdp_checkpointing
from .mixed_precision import bfSixteen, fpSixteen
from .wrapping import get_transformer_wrapper


def get_policies(cfg, model, rank):
    """establish current policies for mixed precision and fsdp wrapping"""

    mixed_precision_policy = None
    wrapping_policy = None

    # mixed precision -----
    if cfg.mixed_precision:
        # bfloat_available = bfloat_support()
        if not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print("FP16 enabled. ")
        else:
            # mixed_precision_policy = policies.fpSixteen
            print(
                "bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    # hardcoded for now
    wrapping_policy = get_transformer_wrapper(model)

    return mixed_precision_policy, wrapping_policy


__all__ = [
    "apply_fsdp_checkpointing",
    "bfSixteen",
    "fpSixteen",
    "get_transformer_wrapper",
    "get_policies",
]