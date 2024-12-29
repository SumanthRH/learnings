# Pytorch Notes

## Table of Contents
- [Helpful links](#helpful-links)
- [FSDP v1](#fsdp-v1)
    - [Mixed Precision](#mixed-precision)
    - [Auto Wrap Policy](#auto-wrap-policy)
    - [State Dict Management](#state-dict-management)
    - [`use_orig_params`](#use_orig_params)
    - [`sync_module_states`](#sync_module_states)
- [FSDP v2](#fsdp-v2)
- [DCP](#dcp)

## Helpful links
- Understanding CUDA Memory Usage:  https://pytorch.org/docs/stable/torch_cuda_memory.html 
- Memory management: https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management 
- Autograd mechanics: https://pytorch.org/docs/stable/notes/autograd.html

## FSDP v1
- FSDP v1:  https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes  

At the heart of FSDP v1 is `FlatParameter` abstraction: this is an atomic unit of communication in FSDP. Each module wrapped in `FullyShardedDataParallel` class (“FSDP”) will have all its parameters combined into one “flat parameter” by FSDP . A single `FlatParameter` is communicated at once during forward/backward pass. Sharded tensors are represented with the `ShardedTensor` API. Note that nesting is allowed i.e you can wrap your entire model in FSDP while also wrapping inner blocks in FSDP, meaning that you can set the granularity for communication. This can also allow FSDP to overlap communication with computation.

### Mixed Precision

AKA please use `torch_dtype=torch.float32` at model init!

FSDP has very fine grained MP settings. You can control computation/parameter dtype, gradient dtype and buffer dtype during communication. You can do something like: 

```python
MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32) 
```

### Auto Wrap Policy

Auto wrap policy deals with how you want to wrap inner modules in FSDP. Why care about this? Because by default, you would wrap your entire model with FSDP, and in v1, this means that FSDP will internally store all parameters as one `FlatParameter` . This is an atomic unit of communication, which means that communication and computation won't overlap during training without any auto wrap policy! 

**For Full Param** 
Simple `transformer_wrap_policy` that wraps each transformer block in FSDP. 

**For LoRA** 
The recommended policy is from the `peft` repo: 
`from peft.utils.other import fsdp_auto_wrap_policy`

This autowrap policy gets around some quirks of working with FSDP and LoRA. It wraps each transformer block in FSDP and within each transformer block, it wraps each lora linear layer individually (i.e one for lora_A and one for lora_B) .
Example: 
```python
FullyShardedDataParallel(
     LlamaDecoderLayer(
      self_attn: LlamaSdpaAttention(
         q_proj: lora.Linear( 
            	base_layer: Linear(...)
		lora_A: FullyShardedDataParallel(...)
		lora_B: FullyShardedDataParallel(...)
	     )
	  ) 
	  ...
      )
)
```

### State Dict Management

You can control how state dicts are collected / loaded during FSDP training
Specify any  `StateDictType` (`FULL_STATE_DICT` or `SHARDED_STATE_DICT`) to get full (gathered) or sharded state dict for checkpointing. For full state dict, can further get the state dict on just rank0 or all ranks. 

Another great feature is that FSDP saves checkpoints with torch DCP by default, which means you get flexible fault tolerant checkpointing for free! 

### `use_orig_params` 

From the docs: 

> Setting this to True has FSDP use module ‘s original parameters. FSDP exposes those original parameters to the user via nn.Module.named_parameters() instead of FSDP’s internal FlatParameter s. This means that the optimizer step runs on the original parameters, enabling per-original-parameter hyperparameters. FSDP preserves the original parameter variables and manipulates their data between unsharded and sharded forms, where they are always views into the underlying unsharded or sharded FlatParameter, respectively. With the current algorithm, the sharded form is always 1D, losing the original tensor structure. An original parameter may have all, some, or none of its data present for a given rank. In the none case, its data will be like a size-0 empty tensor. Users should not author programs relying on what data is present for a given original parameter in its sharded form. True is required to use torch.compile(). Setting this to False exposes FSDP’s internal FlatParameter s to the user via nn.Module.named_parameters()

You should do `use_orig_params=False` for better performance ALL THE TIME. However, `use_orig_params=False` breaks with LoRA depending on the library versions (have seen it break with latest torch and Huggingface ecosystem libraries), so in this case you can stick to `use_orig_params=True`. 


### `sync_module_states` 

sync_module_states=True requires GPU communication - happens when you initialize the FSDP module as FSDP(....) . This will broadcast model weights from rank 0 to the other ranks. This enables loading in a memory efficient way instead of loading a copy in RAM by all ranks. 

In `accelerate`, this is used in sync with `cpu_ram_efficient_loading`, which loads the model only on rank0 and for all other ranks initializes params on meta device. Then, in `accelerate.prepare`, when FSDP modules are initialized, weights are mem copied from rank0 GPU to the respective ranks. For inner workings, see this comment: 
https://github.com/huggingface/accelerate/issues/2100#issuecomment-1822330625  


## FSDP v2

- FSDP v2 API changes(GA in 2.6.0): https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md 
- FSDP 2 RFC in `accelerate` : https://github.com/huggingface/accelerate/pull/3231


## DCP 
- Basic tutorial: https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html 