# Copyright 2025 Qwen-Image Team, The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gaudi-optimized Qwen-Image transformer utilities.

This file is intentionally **minimal** for the tutorial.

It provides only the pieces that we monkey-patch into diffusers:
- `apply_rotary_emb_qwen` (Gaudi-friendly real RoPE math)
- `QwenEmbedRope` (returns cos/sin tensors; avoids complex on HPU)
- `QwenDoubleStreamAttnProcessor2_0` (joint attention using Habana `FusedSDPA`)

Everything not needed for the tutorial (context-parallel, cache, full model classes)
has been removed.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention

try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
except Exception as e:  # pragma: no cover
    raise ImportError("Habana FusedSDPA is required for Gaudi attention patch") from e


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied to `x`.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D//2]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out

    # Gaudi path: freqs_cis is (cos, sin) with shape [S, D//2]
    cos, sin = freqs_cis
    x_pairs = x.reshape(*x.shape[:-1], -1, 2)
    x_real, x_imag = x_pairs.unbind(-1)

    freq_real, freq_imag = cos.unsqueeze(1), sin.unsqueeze(1)  # [S, 1, D//2]

    expand_shape = x_real.shape[1:]
    freq_real_expanded = freq_real.expand(expand_shape).unsqueeze(0)
    freq_imag_expanded = freq_imag.expand(expand_shape).unsqueeze(0)

    out_real = x_real * freq_real_expanded - x_imag * freq_imag_expanded
    out_imag = x_real * freq_imag_expanded + x_imag * freq_real_expanded

    x_out = torch.cat([out_real, out_imag], dim=-1)
    return x_out


class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        # Get cos and sin components separately for positive indices
        pos_cos_0, pos_sin_0 = self.rope_params(pos_index, self.axes_dim[0], self.theta)
        pos_cos_1, pos_sin_1 = self.rope_params(pos_index, self.axes_dim[1], self.theta)
        pos_cos_2, pos_sin_2 = self.rope_params(pos_index, self.axes_dim[2], self.theta)

        self.pos_freqs_cos = torch.cat([pos_cos_0, pos_cos_1, pos_cos_2], dim=1)
        self.pos_freqs_sin = torch.cat([pos_sin_0, pos_sin_1, pos_sin_2], dim=1)

        # Get cos and sin components separately for negative indices
        neg_cos_0, neg_sin_0 = self.rope_params(neg_index, self.axes_dim[0], self.theta)
        neg_cos_1, neg_sin_1 = self.rope_params(neg_index, self.axes_dim[1], self.theta)
        neg_cos_2, neg_sin_2 = self.rope_params(neg_index, self.axes_dim[2], self.theta)

        self.neg_freqs_cos = torch.cat([neg_cos_0, neg_cos_1, neg_cos_2], dim=1)
        self.neg_freqs_sin = torch.cat([neg_sin_0, neg_sin_1, neg_sin_2], dim=1)

        self.rope_cache = {}
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        Returns:
            Tuple of (cos, sin) tensors with real values instead of complex
        """
        assert dim % 2 == 0
        freqs = torch.outer(
            index,
            # 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.bfloat16).div(dim)),
        )
        # Return cos and sin components separately instead of using torch.polar
        cos_freqs = torch.cos(freqs)
        sin_freqs = torch.sin(freqs)
        return cos_freqs, sin_freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args:
            video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video
            txt_seq_lens: [bs] a list of integers representing the length of the text
            device: torch device

        Returns:
            Tuple of (vid_freqs_cos, vid_freqs_sin, txt_freqs_cos, txt_freqs_sin)
        """
        if self.pos_freqs_cos.device != device:
            self.pos_freqs_cos = self.pos_freqs_cos.to(device)
            self.pos_freqs_sin = self.pos_freqs_sin.to(device)
            self.neg_freqs_cos = self.neg_freqs_cos.to(device)
            self.neg_freqs_sin = self.neg_freqs_sin.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs_cos = []
        vid_freqs_sin = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if rope_key not in self.rope_cache:
                self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
            vid_freq_cos, vid_freq_sin = self.rope_cache[rope_key]
            vid_freqs_cos.append(vid_freq_cos)
            vid_freqs_sin.append(vid_freq_sin)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs_cos = self.pos_freqs_cos[max_vid_index : max_vid_index + max_len, ...]
        txt_freqs_sin = self.pos_freqs_sin[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs_cos = torch.cat(vid_freqs_cos, dim=0)
        vid_freqs_sin = torch.cat(vid_freqs_sin, dim=0)

        # Flat tuple for HPU-friendly attention processor
        return vid_freqs_cos, vid_freqs_sin, txt_freqs_cos, txt_freqs_sin

    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos_cos = self.pos_freqs_cos.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_pos_sin = self.pos_freqs_sin.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg_cos = self.neg_freqs_cos.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg_sin = self.neg_freqs_sin.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame_cos = freqs_pos_cos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        freqs_frame_sin = freqs_pos_sin[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)

        if self.scale_rope:
            freqs_height_cos = torch.cat(
                [freqs_neg_cos[1][-(height - height // 2) :], freqs_pos_cos[1][: height // 2]],
                dim=0,
            )
            freqs_height_cos = freqs_height_cos.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_height_sin = torch.cat(
                [freqs_neg_sin[1][-(height - height // 2) :], freqs_pos_sin[1][: height // 2]],
                dim=0,
            )
            freqs_height_sin = freqs_height_sin.view(1, height, 1, -1).expand(frame, height, width, -1)

            freqs_width_cos = torch.cat(
                [freqs_neg_cos[2][-(width - width // 2) :], freqs_pos_cos[2][: width // 2]],
                dim=0,
            )
            freqs_width_cos = freqs_width_cos.view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_width_sin = torch.cat(
                [freqs_neg_sin[2][-(width - width // 2) :], freqs_pos_sin[2][: width // 2]],
                dim=0,
            )
            freqs_width_sin = freqs_width_sin.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height_cos = freqs_pos_cos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_height_sin = freqs_pos_sin[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width_cos = freqs_pos_cos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)
            freqs_width_sin = freqs_pos_sin[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs_cos = torch.cat([freqs_frame_cos, freqs_height_cos, freqs_width_cos], dim=-1).reshape(seq_lens, -1)
        freqs_sin = torch.cat([freqs_frame_sin, freqs_height_sin, freqs_width_sin], dim=-1).reshape(seq_lens, -1)
        return (freqs_cos.clone().contiguous(), freqs_sin.clone().contiguous())


class QwenDoubleStreamAttnProcessor2_0:
    """
    Attention processor for Qwen double-stream architecture, matching DoubleStreamLayerMegatron logic. This processor
    implements joint attention computation where text and image streams are processed together.
    """

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    @torch._dynamo.disable
    def _concat_query_key_value(self, txt_query, txt_key, txt_value, img_query, img_key, img_value):
        return (
            torch.cat([txt_query, img_query], dim=1),
            torch.cat([txt_key, img_key], dim=1),
            torch.cat([txt_value, img_value], dim=1),
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
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

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            vid_cos, vid_sin, txt_cos, txt_sin = image_rotary_emb
            img_freqs = (vid_cos, vid_sin)
            txt_freqs = (txt_cos, txt_sin)

            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query, joint_key, joint_value = self._concat_query_key_value(
            txt_query, txt_key, txt_value, img_query, img_key, img_value
        )

        joint_hidden_states = FusedSDPA.apply(
            joint_query.transpose(1, 2),
            joint_key.transpose(1, 2),
            joint_value.transpose(1, 2),
            None,
            0.0,
            False,
            None,
            "fast",
            None,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.transpose(1, 2).flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
