"""
HPU-friendly monkey patch for Qwen-Image RoPE & attention.

Usage:
    from gaudi_qwen_patch import patch_qwenimage_for_hpu
    pipe = DiffusionPipeline.from_pretrained(..., torch_dtype=torch.bfloat16)
    pipe = patch_qwenimage_for_hpu(pipe, patch_rope=True, patch_fused_attn=False)
    pipe = pipe.to("hpu")
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from diffusers.models.transformers import transformer_qwenimage as tq
import gaudi_transformer_qwenimage as gq


def _patched_call_gaudi_rope_dispatch_attn(
    self,
    attn,
    hidden_states,
    encoder_hidden_states=None,
    encoder_hidden_states_mask=None,
    attention_mask=None,
    image_rotary_emb=None,
):
    if encoder_hidden_states is None:
        raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

    seq_txt = encoder_hidden_states.shape[1]

    img_query = attn.to_q(hidden_states)
    img_key = attn.to_k(hidden_states)
    img_value = attn.to_v(hidden_states)

    txt_query = attn.add_q_proj(encoder_hidden_states)
    txt_key = attn.add_k_proj(encoder_hidden_states)
    txt_value = attn.add_v_proj(encoder_hidden_states)

    img_query = img_query.unflatten(-1, (attn.heads, -1))
    img_key = img_key.unflatten(-1, (attn.heads, -1))
    img_value = img_value.unflatten(-1, (attn.heads, -1))

    txt_query = txt_query.unflatten(-1, (attn.heads, -1))
    txt_key = txt_key.unflatten(-1, (attn.heads, -1))
    txt_value = txt_value.unflatten(-1, (attn.heads, -1))

    if attn.norm_q is not None:
        img_query = attn.norm_q(img_query)
    if attn.norm_k is not None:
        img_key = attn.norm_k(img_key)
    if attn.norm_added_q is not None:
        txt_query = attn.norm_added_q(txt_query)
    if attn.norm_added_k is not None:
        txt_key = attn.norm_added_k(txt_key)

    if image_rotary_emb is not None:
        # gaudi RoPE returns a flat tuple: (vid_cos, vid_sin, txt_cos, txt_sin)
        vid_cos, vid_sin, txt_cos, txt_sin = image_rotary_emb
        img_query = tq.apply_rotary_emb_qwen(img_query, (vid_cos, vid_sin), use_real=False)
        img_key = tq.apply_rotary_emb_qwen(img_key, (vid_cos, vid_sin), use_real=False)
        txt_query = tq.apply_rotary_emb_qwen(txt_query, (txt_cos, txt_sin), use_real=False)
        txt_key = tq.apply_rotary_emb_qwen(txt_key, (txt_cos, txt_sin), use_real=False)

    joint_query = torch.cat([txt_query, img_query], dim=1)
    joint_key = torch.cat([txt_key, img_key], dim=1)
    joint_value = torch.cat([txt_value, img_value], dim=1)

    joint_hidden_states = tq.dispatch_attention_fn(
        joint_query,
        joint_key,
        joint_value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
    )

    joint_hidden_states = joint_hidden_states.flatten(2, 3)
    joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

    txt_attn_output = joint_hidden_states[:, :seq_txt, :]
    img_attn_output = joint_hidden_states[:, seq_txt:, :]

    img_attn_output = attn.to_out[0](img_attn_output)
    if len(attn.to_out) > 1:
        img_attn_output = attn.to_out[1](img_attn_output)

    txt_attn_output = attn.to_add_out(txt_attn_output)

    return img_attn_output, txt_attn_output


def patch_qwenimage_rope_for_hpu(pipe):
    """
    Patch Qwen-Image RoPE using implementations from `gaudi_transformer_qwenimage.py`.

    This replaces:
    - `transformer_qwenimage.QwenEmbedRope` (complex) -> Gaudi real cos/sin RoPE
    - `transformer_qwenimage.apply_rotary_emb_qwen` -> Gaudi version (expects cos/sin, use_real=False)
    - attention processor call -> dispatch attention path, compatible with Gaudi RoPE output format
    """
    tq.QwenEmbedRope = gq.QwenEmbedRope
    tq.apply_rotary_emb_qwen = gq.apply_rotary_emb_qwen
    tq.QwenDoubleStreamAttnProcessor2_0.__call__ = _patched_call_gaudi_rope_dispatch_attn

    # swap existing pos_embed on the pipeline instance (if already constructed)
    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "config"):
        cfg = pipe.transformer.config
        pipe.transformer.pos_embed = gq.QwenEmbedRope(theta=10000, axes_dim=list(cfg.axes_dims_rope), scale_rope=True)

    return pipe


def patch_qwenimage_fused_attn_for_hpu(pipe):
    """
    Patch Qwen-Image joint attention using Gaudi fused-kernel implementation.

    This uses `QwenDoubleStreamAttnProcessor2_0.__call__` from `gaudi_transformer_qwenimage.py`
    (which uses Habana `FusedSDPA`).
    """
    # Ensure RoPE symbols are also patched (fused attention expects Gaudi RoPE format)
    tq.QwenEmbedRope = gq.QwenEmbedRope
    tq.apply_rotary_emb_qwen = gq.apply_rotary_emb_qwen
    # Replace class so instances have helper methods (_concat_query_key_value, etc.)
    tq.QwenDoubleStreamAttnProcessor2_0 = gq.QwenDoubleStreamAttnProcessor2_0

    # Swap existing processors on a loaded pipeline
    if hasattr(pipe, "transformer") and hasattr(pipe.transformer, "transformer_blocks"):
        for blk in pipe.transformer.transformer_blocks:
            if hasattr(blk, "attn") and hasattr(blk.attn, "processor"):
                blk.attn.processor = gq.QwenDoubleStreamAttnProcessor2_0()
    return pipe


def patch_qwenimage_for_hpu(pipe, patch_rope: bool = True, patch_fused_attn: bool = False):
    """
    Apply HPU monkey patches to a loaded DiffusionPipeline.

    - `patch_rope`: replace complex RoPE with real cos/sin implementation.
    - `patch_fused_attn`: replace joint attention compute with Habana FusedSDPA (if available).
    """
    if patch_rope:
        pipe = patch_qwenimage_rope_for_hpu(pipe)

    if patch_fused_attn:
        pipe = patch_qwenimage_fused_attn_for_hpu(pipe)

    return pipe


__all__ = [
    "patch_qwenimage_for_hpu",
    "patch_qwenimage_rope_for_hpu",
    "patch_qwenimage_fused_attn_for_hpu",
]

