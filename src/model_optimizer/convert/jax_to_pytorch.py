# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# OpenPI JAX (Orbax) checkpoint → PyTorch ``PI0Pytorch`` + ``model.safetensors``。
# 逻辑源自 openpi ``examples/convert_jax_model_to_pytorch.py``；配置解析与
# ``serve_policy`` 一致（``load_train_config``：注册名或 ``TrainConfig`` 的 ``.py``）。

import json
import logging
import os
import pathlib
import shutil
from typing import Any, Literal

from flax import traverse_util
from flax.nnx import traversals
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
from openpi.training import utils

from model_optimizer.openpi_train_config import load_train_config

logger = logging.getLogger(__name__)


def _parse_restore_dtype(dtype: str | jnp.dtype | np.dtype | None):
    if dtype is None:
        return None
    if isinstance(dtype, str):
        return np.dtype(dtype)
    return dtype


def _orbax_restore_item_from_metadata(metadata: Any) -> dict[str, Any]:
    """Build ``{"params": <pytree>}`` for ``PyTreeRestore`` from orbax metadata.

    Older orbax returns a dict (``metadata["params"]``). Newer orbax returns
    ``StepMetadata`` with ``item_metadata`` (see orbax-checkpoint >= 0.12).
    """
    try:
        return {"params": metadata["params"]}
    except TypeError:
        pass

    item_metadata = getattr(metadata, "item_metadata", None)
    if item_metadata is None:
        raise TypeError(
            f"Unsupported orbax metadata type {type(metadata)!r}. "
            "Install openpi-pinned orbax: uv pip install 'orbax-checkpoint==0.11.13'"
        ) from None

    if isinstance(item_metadata, dict):
        params_meta = item_metadata.get("params", item_metadata)
    else:
        try:
            params_meta = item_metadata["params"]
        except (TypeError, KeyError):
            params_meta = getattr(item_metadata, "params", item_metadata)

    if hasattr(params_meta, "as_nested_tree"):
        params_tree = params_meta.as_nested_tree()
    elif isinstance(params_meta, dict):
        params_tree = params_meta
    else:
        params_tree = params_meta

    return {"params": params_tree}


def restore_orbax_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: str | jnp.dtype | np.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> dict[str, Any]:
    """Restore OpenPI ``params/`` subtree; compatible with old and new orbax metadata."""
    params_path = (
        pathlib.Path(params_path).resolve()
        if not str(params_path).startswith("gs://")
        else params_path
    )
    dtype = _parse_restore_dtype(dtype)

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    try:
        return openpi.models.model.restore_params(
            params_path, restore_type=restore_type, dtype=dtype, sharding=sharding
        )
    except TypeError as exc:
        if "subscriptable" not in str(exc) and "StepMetadata" not in str(exc):
            raise
        logger.warning(
            "openpi restore_params failed (%s); using orbax StepMetadata-compatible restore",
            exc,
        )

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = _orbax_restore_item_from_metadata(metadata)
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=restore_type, dtype=dtype
                    ),
                    item,
                ),
            ),
        )["params"]

    flat_params = traverse_util.flatten_dict(params)
    if flat_params and all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(
        -1, config.vision_config.hidden_size
    )

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(
        f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}"
    )
    encoderblock_layernorm0_bias = state_dict.pop(
        f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}"
    )
    encoderblock_layernorm1_scale = state_dict.pop(
        f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}"
    )
    encoderblock_layernorm1_bias = state_dict.pop(
        f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}"
    )

    encoderblock_mlp_dense0_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}"
    )
    encoderblock_mlp_dense0_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}"
    )
    encoderblock_mlp_dense1_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}"
    )
    encoderblock_mlp_dense1_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}"
    )

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = (
            encoderblock_attention_0_key_kernel[i]
            .reshape(-1, config.vision_config.hidden_size)
            .transpose()
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = (
            encoderblock_attention_0_key_bias[i]
            .reshape(-1, config.vision_config.hidden_size)
            .reshape(-1)
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = (
            encoderblock_attention_0_value_kernel[i]
            .reshape(-1, config.vision_config.hidden_size)
            .transpose()
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = (
            encoderblock_attention_0_value_bias[i]
            .reshape(-1, config.vision_config.hidden_size)
            .reshape(-1)
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = (
            encoderblock_attention_0_query_kernel[i]
            .reshape(-1, config.vision_config.hidden_size)
            .transpose()
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = (
            encoderblock_attention_0_query_bias[i]
            .reshape(-1, config.vision_config.hidden_size)
            .reshape(-1)
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = (
            encoderblock_attention_0_out_kernel[i]
            .reshape(-1, config.vision_config.hidden_size)
            .transpose()
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = (
            encoderblock_attention_0_out_bias[i]
            .reshape(-1, config.vision_config.hidden_size)
            .reshape(-1)
        )

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    )
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    )
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma)
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    )
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # pop the einsum attention + mlp representations
    llm_attention_attn_vec_einsum = state_dict.pop(
        f"llm/layers/attn/attn_vec_einsum/w{suffix}"
    )
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(
        f"llm/layers/pre_ffw_norm/scale{suffix}"
    )

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim,
                config.text_config.hidden_size,
            )
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"
        ] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"
        ] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"
        ] = v_proj_weight_reshaped

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim,
                config.text_config.hidden_size,
            )
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"
        ] = o_proj_weight_reshaped

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"
        ] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"
        ] = up_proj_weight.transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"
        ] = llm_mlp_linear[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"
        ] = llm_input_layernorm[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}

    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, pi05: bool):
    """Convert Gemma JAX parameters to PyTorch format."""
    # Add missing attributes to config if they don't exist
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = (
        "/value"
        if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict
        else ""
    )

    llm_attention_attn_vec_einsum = state_dict.pop(
        f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}"
    )
    llm_attention_kv_einsum = state_dict.pop(
        f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}"
    )
    llm_attention_q_einsum = state_dict.pop(
        f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}"
    )

    llm_mlp_gating_einsum = state_dict.pop(
        f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}"
    )
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Pi05 uses adaptive norm (Dense); Pi0 uses RMSNorm (scale).
    if pi05:
        # Pi05 with adaptive normalization
        llm_input_layernorm_bias = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}"
        )
        llm_post_attention_layernorm_bias = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}"
        )
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular pi0 with standard RMSNorm
        llm_input_layernorm = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}"
        )
        llm_post_attention_layernorm = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}"
        )

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"
        ] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"
        ] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"
        ] = v_proj_weight_reshaped

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"
        ] = o_proj_weight_reshaped

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"
        ] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"
        ] = up_proj_weight.transpose()
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"
        ] = llm_mlp_linear[i].transpose()

        if pi05:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"
            ] = llm_input_layernorm_bias[i]
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"
            ] = llm_post_attention_layernorm_bias[i]
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"
            ] = llm_input_layernorm_kernel[i].transpose()
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"
            ] = llm_post_attention_layernorm_kernel[i].transpose()
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"
            ] = llm_input_layernorm[i]
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"
            ] = llm_post_attention_layernorm[i]

    # Handle final norm layer
    if pi05:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(
            f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}"
        )
        final_norm_kernel = state_dict.pop(
            f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = (
            final_norm_bias
        )
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = (
            final_norm_kernel.transpose()
        )
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = (
            state_dict.pop(f"llm/final_norm_{num_expert}/scale{suffix}")
        )

        # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # weights are tied.

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def slice_initial_orbax_checkpoint(
    checkpoint_dir: str, restore_precision: str | None = None
):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    """
    params = restore_orbax_params(
        f"{checkpoint_dir}/params/",
        restore_type=np.ndarray,
        dtype=restore_precision,
    )

    return {
        "paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"),
        "projection_params": params,
    }


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.

    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    checkpoint_dir = (
        os.path.abspath(checkpoint_dir)
        if not checkpoint_dir.startswith("gs://")
        else checkpoint_dir
    )
    params = restore_orbax_params(f"{checkpoint_dir}/params/", restore_type=np.ndarray)
    print(utils.array_tree_to_info(params))


def convert_pi0_checkpoint(
    checkpoint_dir: str,
    precision: str,
    output_path: str,
    model_config: openpi.models.pi0_config.Pi0Config,
):
    """
    Convert PI0 JAX checkpoint to PyTorch format.

    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16, float16)
        output_path: Path to save the converted PyTorch model
        model_config: Model config
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")

    # Break down orbax ckpts by restoring via JAX to respect dtype
    initial_params = slice_initial_orbax_checkpoint(
        checkpoint_dir=checkpoint_dir, restore_precision="float32"
    )

    # Process projection params
    if model_config.pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"

        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # Create configs based on checkpoint path
    # All models use the same PaliGemma config structure
    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_config = PaliGemmaConfig()
    action_expert_config = openpi.models.gemma.get_config("gemma_300m")

    # Process PaliGemma weights
    paligemma_params, expert_params = slice_paligemma_state_dict(
        initial_params["paligemma_params"], paligemma_config
    )

    # Process Gemma weights from expert_params
    gemma_params = slice_gemma_state_dict(
        expert_params,
        action_expert_config,
        num_expert=1,
        pi05=model_config.pi05,
    )

    # Instantiate model
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)

    # Combine all parameters (no prefix needed for our model structure)
    all_params = {**paligemma_params, **gemma_params, **projection_params}

    # Load state dict
    pi0_model.load_state_dict(all_params, strict=False)

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    # Save the converted model using safetensors
    os.makedirs(output_path, exist_ok=True)

    # Save model weights as SafeTensors using save_model to handle tied weights
    safetensors.torch.save_model(
        pi0_model, os.path.join(output_path, "model.safetensors")
    )

    # Copy assets folder if it exists
    assets_source = pathlib.Path(checkpoint_dir).parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        shutil.copytree(assets_source, assets_dest)

    # Save config as JSON for reference
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def resolve_pi0_model_config(config_ref: str) -> openpi.models.pi0_config.Pi0Config:
    """``load_train_config`` → ``TrainConfig.model``，且必须为 ``Pi0Config``。"""
    model_config = load_train_config(config_ref).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_ref!r} is not a Pi0Config (got {type(model_config)!r})")
    return model_config


def run_jax_to_pytorch(
    checkpoint_dir: str,
    config_ref: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
) -> None:
    """按训练配置将 JAX checkpoint 转为 PyTorch，或仅打印参数树。"""
    model_config = resolve_pi0_model_config(config_ref)
    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
        return
    if not output_path:
        raise ValueError(
            "output_path is required for conversion (omit --inspect-only and set --output-path)"
        )
    convert_pi0_checkpoint(checkpoint_dir, precision, output_path, model_config)
