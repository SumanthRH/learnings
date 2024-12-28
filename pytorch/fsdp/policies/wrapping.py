# holds various wrapping policies for fsdp


import functools

from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.models.t5.modeling_t5 import T5Block


def get_module_class_from_name(module, name):
    """
    Gets a class from a module by its name.

    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


def get_size_policy(min_params=1e8):
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


def get_transformer_wrapper(model):
    """we register our main layer class and use the fsdp transformer wrapping policy
    ensures embedding layers are in the root fsdp unit for shared access and that fsdp units map to transformer layers
    """
    # ====   use new transformer wrapper
    no_split_modules = getattr(model, "_no_split_modules", None)
    default_transformer_cls_names_to_wrap = (
        list(no_split_modules) if no_split_modules is not None else []
    )
    if not len(default_transformer_cls_names_to_wrap):
        raise NotImplementedError(
            "Warning: Could not detect transformer classes to wrap"
        )
    transformer_cls_to_wrap = set()
    for layer_class in default_transformer_cls_names_to_wrap:
        transformer_cls = get_module_class_from_name(model, layer_class)
        if transformer_cls is None:
            raise ValueError(
                f"Could not find the transformer layer class {layer_class} in the model."
            )
        transformer_cls_to_wrap.add(transformer_cls)
    print(f"Wrapping {transformer_cls_to_wrap}")
    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls_to_wrap,
    )

    return t5_auto_wrap_policy