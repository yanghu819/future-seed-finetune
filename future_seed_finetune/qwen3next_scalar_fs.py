from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qwen35_scalar_fs import FutureSeedRuntime, ScalarFutureSeedConfig, _PATCHED, _RUNTIME, _prepare_seed


_PATCHED_NEXT = False


def _resolve_qwen3next_symbols() -> dict[str, Any]:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Cache,
        Qwen3NextDecoderLayer,
        Qwen3NextForCausalLM,
        Qwen3NextGatedDeltaNet,
        Qwen3NextModel,
        apply_mask_to_padding_states,
    )

    return {
        "Cache": Cache,
        "Qwen3NextDecoderLayer": Qwen3NextDecoderLayer,
        "Qwen3NextForCausalLM": Qwen3NextForCausalLM,
        "Qwen3NextGatedDeltaNet": Qwen3NextGatedDeltaNet,
        "Qwen3NextModel": Qwen3NextModel,
        "apply_mask_to_padding_states": apply_mask_to_padding_states,
    }


def install_qwen3next_scalar_fs_patch() -> None:
    global _PATCHED_NEXT
    if _PATCHED_NEXT:
        return

    symbols = _resolve_qwen3next_symbols()
    Qwen3NextGatedDeltaNet = symbols["Qwen3NextGatedDeltaNet"]
    Qwen3NextDecoderLayer = symbols["Qwen3NextDecoderLayer"]
    Qwen3NextModel = symbols["Qwen3NextModel"]
    apply_mask_to_padding_states = symbols["apply_mask_to_padding_states"]

    original_gated_forward = Qwen3NextGatedDeltaNet.forward
    original_decoder_forward = Qwen3NextDecoderLayer.forward
    original_model_forward = Qwen3NextModel.forward

    def patched_gated_forward(self, hidden_states, cache_params=None, attention_mask=None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1).transpose(1, 2)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        fs_cfg = getattr(self, "_future_seed_config", None)
        fs_selected = bool(getattr(self, "_future_seed_selected", False))
        fs_active = bool(fs_cfg and fs_selected and seq_len > 1)
        incoming_seed = getattr(self, "_future_seed_initial_state_override", None)
        prepared_seed = _prepare_seed(incoming_seed, self, fs_cfg) if fs_active else None

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=prepared_seed,
                output_final_state=(cache_params is not None) or fs_active,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        self._future_seed_last_recurrent_state = last_recurrent_state if fs_active else None
        self._future_seed_used_input_state = prepared_seed is not None

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)
        return self.out_proj(core_attn_out)

    def patched_decoder_forward(self, *args, **kwargs):
        runtime = _RUNTIME.get()
        if runtime is not None and getattr(runtime, "active", False) and self.layer_type == "linear_attention":
            if getattr(self.linear_attn, "_future_seed_selected", False):
                self.linear_attn._future_seed_initial_state_override = runtime.current_seed
                runtime.previous_seed_present = runtime.current_seed is not None
            else:
                self.linear_attn._future_seed_initial_state_override = None

        output = original_decoder_forward(self, *args, **kwargs)

        if runtime is not None and getattr(runtime, "active", False):
            if self.layer_type == "linear_attention":
                if getattr(self.linear_attn, "_future_seed_selected", False):
                    runtime.current_seed = getattr(self.linear_attn, "_future_seed_last_recurrent_state", None)
                    runtime.captured_layers.append(self.linear_attn.layer_idx)
                    if getattr(self.linear_attn, "_future_seed_used_input_state", False):
                        runtime.injection_count += 1
                        runtime.injected_layers.append(self.linear_attn.layer_idx)
                else:
                    runtime.current_seed = None
            elif self.layer_type == "full_attention" and runtime.config.reset_on_full_attention:
                runtime.current_seed = None

        return output

    def patched_model_forward(self, *args, **kwargs):
        fs_cfg = getattr(self, "_future_seed_config", None)
        if fs_cfg is None or not fs_cfg.enabled:
            outputs = original_model_forward(self, *args, **kwargs)
            self._future_seed_last_runtime = None
            return outputs

        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]
        inputs_embeds = kwargs.get("inputs_embeds")

        seq_len = None
        if input_ids is not None:
            seq_len = input_ids.shape[1]
        elif inputs_embeds is not None:
            seq_len = inputs_embeds.shape[1]

        active = seq_len is not None and (not fs_cfg.prompt_only or seq_len > 1)
        runtime = FutureSeedRuntime(active=bool(active), config=fs_cfg)
        token = _RUNTIME.set(runtime)
        try:
            outputs = original_model_forward(self, *args, **kwargs)
            self._future_seed_last_runtime = runtime.summary()
            return outputs
        finally:
            _RUNTIME.reset(token)

    Qwen3NextGatedDeltaNet.forward = patched_gated_forward
    Qwen3NextDecoderLayer.forward = patched_decoder_forward
    Qwen3NextModel.forward = patched_model_forward
    Qwen3NextGatedDeltaNet._future_seed_original_forward = original_gated_forward
    Qwen3NextDecoderLayer._future_seed_original_forward = original_decoder_forward
    Qwen3NextModel._future_seed_original_forward = original_model_forward
    _PATCHED_NEXT = True


def apply_qwen3next_scalar_future_seed(model: nn.Module, cfg: ScalarFutureSeedConfig) -> nn.Module:
    install_qwen3next_scalar_fs_patch()
    backbone = getattr(model, "model", model)
    backbone._future_seed_config = cfg

    for idx, layer in enumerate(backbone.layers):
        if getattr(layer, "layer_type", None) == "linear_attention" and idx >= cfg.start_layer:
            if not hasattr(layer.linear_attn, "fs_alpha"):
                layer.linear_attn.register_parameter(
                    "fs_alpha",
                    nn.Parameter(torch.tensor(float(cfg.alpha_init), dtype=torch.float32)),
                )
            layer.linear_attn._future_seed_config = cfg
            layer.linear_attn._future_seed_selected = True
        elif getattr(layer, "layer_type", None) == "linear_attention":
            layer.linear_attn._future_seed_config = cfg
            layer.linear_attn._future_seed_selected = False
    return model

