import gc
import os
from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.distributed.utils import _free_storage


_SHARDING_STRATEGIES = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
}


def get_sharding_strategy():
    name = os.getenv("FSDP_SHARDING_STRATEGY", "SHARD_GRAD_OP").strip().upper()
    try:
        return _SHARDING_STRATEGIES[name]
    except KeyError as exc:
        supported = ", ".join(_SHARDING_STRATEGIES)
        raise ValueError(
            f"Unsupported FSDP_SHARDING_STRATEGY={name!r}; expected one of: {supported}"
        ) from exc


def shard_model(
    model,
    device_id,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
    process_group=None,
    sharding_strategy=None,
    sync_module_states=True,
    use_lora=False
):
    if sharding_strategy is None:
        sharding_strategy = get_sharding_strategy()
    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype),
        device_id=device_id,
        sync_module_states=sync_module_states,
        use_orig_params=True if use_lora else False)
    return model


def free_model(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            _free_storage(m._handle.flat_param.data)
    del model
    gc.collect()
    torch.cuda.empty_cache()
