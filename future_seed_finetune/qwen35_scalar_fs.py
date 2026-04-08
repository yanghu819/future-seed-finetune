from __future__ import annotations

import json
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


_RUNTIME: ContextVar["FutureSeedRuntime | None"] = ContextVar("future_seed_runtime", default=None)
_PATCHED = False


@dataclass
class ScalarFutureSeedConfig:
    enabled: bool = True
    start_layer: int = 0
    prompt_only: bool = True
    detach_seed: bool = True
    rms_norm_seed: bool = True
    clip_value: float | None = 1.0
    alpha_init: float = 0.0
    enable_delta_adapter: bool = False
    seed_projector_rank: int = 0
    projection_lora_rank: int = 0
    projection_lora_alpha: float = 1.0
    reset_on_full_attention: bool = True


@dataclass
class FutureSeedRuntime:
    active: bool
    config: ScalarFutureSeedConfig
    prompt_length: int | None = None
    injection_count: int = 0
    captured_layers: list[int] | None = None
    injected_layers: list[int] | None = None
    previous_seed_present: bool = False
    current_seed: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.captured_layers is None:
            self.captured_layers = []
        if self.injected_layers is None:
            self.injected_layers = []

    def summary(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "config": asdict(self.config),
            "prompt_length": self.prompt_length,
            "injection_count": self.injection_count,
            "captured_layers": list(self.captured_layers or []),
            "injected_layers": list(self.injected_layers or []),
            "previous_seed_present": bool(self.previous_seed_present),
        }


def _resolve_qwen35_symbols() -> dict[str, Any]:
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
    )
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Cache,
        Qwen3_5ForConditionalGeneration,
        Qwen3_5DecoderLayer,
        Qwen3_5ForCausalLM,
        Qwen3_5GatedDeltaNet,
        Qwen3_5TextModel,
        Qwen3_5TextRotaryEmbedding,
        apply_mask_to_padding_states,
    )

    return {
        "Cache": Cache,
        "Qwen3_5Config": Qwen3_5Config,
        "Qwen3_5ForConditionalGeneration": Qwen3_5ForConditionalGeneration,
        "Qwen3_5DecoderLayer": Qwen3_5DecoderLayer,
        "Qwen3_5ForCausalLM": Qwen3_5ForCausalLM,
        "Qwen3_5GatedDeltaNet": Qwen3_5GatedDeltaNet,
        "Qwen3_5TextConfig": Qwen3_5TextConfig,
        "Qwen3_5TextModel": Qwen3_5TextModel,
        "Qwen3_5TextRotaryEmbedding": Qwen3_5TextRotaryEmbedding,
        "apply_mask_to_padding_states": apply_mask_to_padding_states,
    }


def normalize_qwen35_text_config(config: Any) -> Any:
    mlp_only_layers = getattr(config, "mlp_only_layers", None)
    if isinstance(mlp_only_layers, AttributeError) or mlp_only_layers is None:
        config.mlp_only_layers = []
    else:
        config.mlp_only_layers = list(mlp_only_layers)
    return config


def load_qwen35_text_config(model_dir: str | Path) -> Any:
    symbols = _resolve_qwen35_symbols()
    Qwen3_5TextConfig = symbols["Qwen3_5TextConfig"]
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    text_config_dict = raw.get("text_config", raw)
    config = Qwen3_5TextConfig.from_dict(text_config_dict)
    return normalize_qwen35_text_config(config)


def normalize_qwen35_full_config(config: Any) -> Any:
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        config.text_config = normalize_qwen35_text_config(text_config)

    vision_config = getattr(config, "vision_config", None)
    if vision_config is not None:
        deepstack_visual_indexes = getattr(vision_config, "deepstack_visual_indexes", None)
        if isinstance(deepstack_visual_indexes, AttributeError) or deepstack_visual_indexes is None:
            vision_config.deepstack_visual_indexes = []
        else:
            vision_config.deepstack_visual_indexes = list(deepstack_visual_indexes)
        config.vision_config = vision_config
    return config


def load_qwen35_full_config(model_dir: str | Path) -> Any:
    symbols = _resolve_qwen35_symbols()
    Qwen3_5Config = symbols["Qwen3_5Config"]
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    config = Qwen3_5Config.from_dict(raw)
    return normalize_qwen35_full_config(config)


def detect_qwen35_pretrained_architecture(model_dir: str | Path) -> str:
    raw = json.loads((Path(model_dir) / "config.json").read_text())
    architectures = set(raw.get("architectures", []))
    if "Qwen3_5ForConditionalGeneration" in architectures:
        return "conditional_generation"
    return "causal_lm"


def _prepare_seed(seed: torch.Tensor | None, module: nn.Module, cfg: ScalarFutureSeedConfig) -> torch.Tensor | None:
    if seed is None:
        return None

    out = seed.detach() if cfg.detach_seed else seed
    out = out.to(device=module.A_log.device, dtype=torch.float32)

    if cfg.rms_norm_seed:
        reduce_dims = tuple(range(1, out.ndim))
        rms = out.pow(2).mean(dim=reduce_dims, keepdim=True).sqrt().clamp_min(1e-6)
        out = out / rms

    out = out * module.fs_alpha.float()

    if cfg.clip_value is not None:
        out = out.clamp(min=-cfg.clip_value, max=cfg.clip_value)

    return out.to(dtype=module.A_log.dtype)


def _apply_seed_projector(
    prepared_seed: torch.Tensor | None,
    hidden_states: torch.Tensor,
    module: nn.Module,
    cfg: ScalarFutureSeedConfig,
) -> torch.Tensor | None:
    if prepared_seed is None or cfg.seed_projector_rank <= 0:
        return prepared_seed
    if getattr(module, "seed_proj_in", None) is None:
        return prepared_seed

    seed_float = prepared_seed.float()
    projected = torch.einsum("bhvk,kr->bhvr", seed_float, module.seed_proj_in.float())
    projected = torch.einsum("bhvr,rk->bhvk", projected, module.seed_proj_out.float())

    prompt_summary = hidden_states.float().mean(dim=1)
    gate = torch.sigmoid(
        F.linear(
            prompt_summary,
            module.seed_gate_vector.float().unsqueeze(0),
            module.seed_gate_bias.float().view(1),
        )
    )
    out = seed_float + gate.view(gate.shape[0], 1, 1, 1) * projected
    if cfg.clip_value is not None:
        out = out.clamp(min=-cfg.clip_value, max=cfg.clip_value)
    return out.to(dtype=module.A_log.dtype)


def _apply_projection_lora(
    hidden_states: torch.Tensor,
    module: nn.Module,
    prefix: str,
    cfg: ScalarFutureSeedConfig,
) -> torch.Tensor | None:
    if cfg.projection_lora_rank <= 0:
        return None
    lora_a = getattr(module, f"{prefix}_lora_A", None)
    lora_b = getattr(module, f"{prefix}_lora_B", None)
    if lora_a is None or lora_b is None:
        return None

    scale = float(cfg.projection_lora_alpha) / float(cfg.projection_lora_rank)
    hidden_float = hidden_states.float()
    delta = F.linear(hidden_float, lora_a.float())
    delta = F.linear(delta, lora_b.float())
    return (delta * scale).to(dtype=hidden_states.dtype)


def install_qwen35_upstream_compat_fixes() -> None:
    symbols = _resolve_qwen35_symbols()
    Qwen3_5TextRotaryEmbedding = symbols["Qwen3_5TextRotaryEmbedding"]

    if not getattr(Qwen3_5TextRotaryEmbedding, "_future_seed_fixed_init", False):
        def fixed_compute_default_rope_parameters(config=None, device=None, seq_len=None):
            base = config.rope_parameters["rope_theta"]
            partial_rotary_factor = config.rope_parameters.get("partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
            )
            return inv_freq, 1.0

        def fixed_init(self, config, device=None):
            Qwen3_5TextRotaryEmbedding.compute_default_rope_parameters = staticmethod(fixed_compute_default_rope_parameters)
            nn.Module.__init__(self)
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            self.config = config
            self.rope_type = self.config.rope_parameters["rope_type"]
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type != "default":
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
            self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

        Qwen3_5TextRotaryEmbedding.__init__ = fixed_init
        Qwen3_5TextRotaryEmbedding._future_seed_fixed_init = True


def install_qwen35_scalar_fs_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    install_qwen35_upstream_compat_fixes()
    from .qwen3next_scalar_fs import install_qwen3next_scalar_fs_patch

    install_qwen3next_scalar_fs_patch()
    symbols = _resolve_qwen35_symbols()
    Qwen3_5GatedDeltaNet = symbols["Qwen3_5GatedDeltaNet"]
    Qwen3_5DecoderLayer = symbols["Qwen3_5DecoderLayer"]
    Qwen3_5TextModel = symbols["Qwen3_5TextModel"]
    apply_mask_to_padding_states = symbols["apply_mask_to_padding_states"]

    original_gated_forward = Qwen3_5GatedDeltaNet.forward
    original_decoder_forward = Qwen3_5DecoderLayer.forward
    original_text_model_forward = Qwen3_5TextModel.forward

    def patched_text_model_forward(self, *args, **kwargs):
        fs_cfg = getattr(self, "_future_seed_config", None)
        if fs_cfg is None or not fs_cfg.enabled:
            outputs = original_text_model_forward(self, *args, **kwargs)
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
        prompt_length = getattr(self, "_future_seed_prompt_length_override", None)
        runtime = FutureSeedRuntime(active=bool(active), config=fs_cfg, prompt_length=prompt_length)
        token = _RUNTIME.set(runtime)
        try:
            outputs = original_text_model_forward(self, *args, **kwargs)
            self._future_seed_last_runtime = runtime.summary()
            return outputs
        finally:
            _RUNTIME.reset(token)

    def patched_gated_forward(self, hidden_states, cache_params=None, attention_mask=None):
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        fs_cfg = getattr(self, "_future_seed_config", None)
        fs_selected = bool(getattr(self, "_future_seed_selected", False))
        fs_active = bool(fs_cfg and fs_selected and seq_len > 1)
        incoming_seed = getattr(self, "_future_seed_initial_state_override", None)
        runtime = _RUNTIME.get()
        runtime_prompt_len = getattr(runtime, "prompt_length", None) if runtime is not None else None

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z_hidden = self.in_proj_z(hidden_states)
        z_lora = _apply_projection_lora(hidden_states, self, "z", fs_cfg) if fs_cfg else None
        if z_lora is not None:
            z_hidden = z_hidden + z_lora
        z = z_hidden.reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

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
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        prepared_seed = _prepare_seed(incoming_seed, self, fs_cfg) if fs_active else None
        if fs_active:
            prepared_seed = _apply_seed_projector(prepared_seed, hidden_states, self, fs_cfg)

        if not use_precomputed_states:
            if fs_active and runtime_prompt_len is not None and 0 < int(runtime_prompt_len) < seq_len:
                prompt_len = int(runtime_prompt_len)
                prompt_attn_out, prompt_final_state = self.chunk_gated_delta_rule(
                    query[:, :prompt_len],
                    key[:, :prompt_len],
                    value[:, :prompt_len],
                    g=g[:, :prompt_len],
                    beta=beta[:, :prompt_len],
                    initial_state=prepared_seed,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                answer_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    query[:, prompt_len:],
                    key[:, prompt_len:],
                    value[:, prompt_len:],
                    g=g[:, prompt_len:],
                    beta=beta[:, prompt_len:],
                    initial_state=prompt_final_state,
                    output_final_state=(cache_params is not None) or fs_active,
                    use_qk_l2norm_in_kernel=True,
                )
                core_attn_out = torch.cat([prompt_attn_out, answer_attn_out], dim=1)
                seed_recurrent_state = prompt_final_state
            else:
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
                seed_recurrent_state = last_recurrent_state
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
            seed_recurrent_state = last_recurrent_state

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        self._future_seed_last_recurrent_state = seed_recurrent_state if fs_active else None
        self._future_seed_used_input_state = prepared_seed is not None

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        out_input = core_attn_out
        core_attn_out = self.out_proj(out_input)
        out_lora = _apply_projection_lora(out_input, self, "out", fs_cfg) if fs_cfg else None
        if out_lora is not None:
            core_attn_out = core_attn_out + out_lora
        if fs_active and getattr(self, "delta_gain", None) is not None:
            gain = 1.0 + self.delta_gain.to(dtype=core_attn_out.dtype).view(1, 1, -1)
            bias = self.delta_bias.to(dtype=core_attn_out.dtype).view(1, 1, -1)
            core_attn_out = core_attn_out * gain + bias
        return core_attn_out

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

    Qwen3_5GatedDeltaNet.forward = patched_gated_forward
    Qwen3_5DecoderLayer.forward = patched_decoder_forward
    Qwen3_5TextModel.forward = patched_text_model_forward
    Qwen3_5GatedDeltaNet._future_seed_original_forward = original_gated_forward
    Qwen3_5DecoderLayer._future_seed_original_forward = original_decoder_forward
    Qwen3_5TextModel._future_seed_original_forward = original_text_model_forward
    _PATCHED = True


def _get_text_backbone(model: nn.Module) -> nn.Module:
    if hasattr(model, "language_model"):
        return model.language_model
    core = getattr(model, "model", None)
    if core is not None and hasattr(core, "language_model"):
        return core.language_model
    return getattr(model, "model", model)


def apply_scalar_future_seed(model: nn.Module, cfg: ScalarFutureSeedConfig) -> nn.Module:
    install_qwen35_scalar_fs_patch()
    backbone = _get_text_backbone(model)
    backbone._future_seed_config = cfg

    for idx, layer in enumerate(backbone.layers):
        if getattr(layer, "layer_type", None) == "linear_attention" and idx >= cfg.start_layer:
            if not hasattr(layer.linear_attn, "fs_alpha"):
                layer.linear_attn.register_parameter(
                    "fs_alpha",
                    nn.Parameter(
                        torch.tensor(
                            float(cfg.alpha_init),
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
            if cfg.enable_delta_adapter and not hasattr(layer.linear_attn, "delta_gain"):
                hidden_size = int(layer.linear_attn.out_proj.out_features)
                layer.linear_attn.register_parameter(
                    "delta_gain",
                    nn.Parameter(
                        torch.zeros(
                            hidden_size,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
                layer.linear_attn.register_parameter(
                    "delta_bias",
                    nn.Parameter(
                        torch.zeros(
                            hidden_size,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
            if cfg.seed_projector_rank > 0 and not hasattr(layer.linear_attn, "seed_proj_in"):
                state_dim = int(layer.linear_attn.head_k_dim)
                hidden_size = int(layer.linear_attn.out_proj.out_features)
                rank = int(cfg.seed_projector_rank)
                layer.linear_attn.register_parameter(
                    "seed_proj_in",
                    nn.Parameter(
                        torch.zeros(
                            state_dim,
                            rank,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
                layer.linear_attn.register_parameter(
                    "seed_proj_out",
                    nn.Parameter(
                        torch.zeros(
                            rank,
                            state_dim,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
                layer.linear_attn.register_parameter(
                    "seed_gate_vector",
                    nn.Parameter(
                        torch.zeros(
                            hidden_size,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
                layer.linear_attn.register_parameter(
                    "seed_gate_bias",
                    nn.Parameter(
                        torch.zeros(
                            1,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
            if cfg.projection_lora_rank > 0 and not hasattr(layer.linear_attn, "z_lora_A"):
                rank = int(cfg.projection_lora_rank)
                z_in = int(layer.linear_attn.in_proj_z.in_features)
                z_out = int(layer.linear_attn.in_proj_z.out_features)
                out_in = int(layer.linear_attn.out_proj.in_features)
                out_out = int(layer.linear_attn.out_proj.out_features)

                layer.linear_attn.register_parameter(
                    "z_lora_A",
                    nn.Parameter(
                        torch.randn(
                            rank,
                            z_in,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                        * 0.01
                    ),
                )
                layer.linear_attn.register_parameter(
                    "z_lora_B",
                    nn.Parameter(
                        torch.zeros(
                            z_out,
                            rank,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
                layer.linear_attn.register_parameter(
                    "out_lora_A",
                    nn.Parameter(
                        torch.randn(
                            rank,
                            out_in,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                        * 0.01
                    ),
                )
                layer.linear_attn.register_parameter(
                    "out_lora_B",
                    nn.Parameter(
                        torch.zeros(
                            out_out,
                            rank,
                            dtype=torch.float32,
                            device=layer.linear_attn.out_proj.weight.device,
                        )
                    ),
                )
            layer.linear_attn._future_seed_config = cfg
            layer.linear_attn._future_seed_selected = True
        elif getattr(layer, "layer_type", None) == "linear_attention":
            layer.linear_attn._future_seed_config = cfg
            layer.linear_attn._future_seed_selected = False

    return model


def freeze_except_future_seed(model: nn.Module) -> list[str]:
    trainable: list[str] = []
    for name, param in model.named_parameters():
        param.requires_grad = (
            name.endswith("fs_alpha")
            or name.endswith("delta_gain")
            or name.endswith("delta_bias")
            or name.endswith("seed_proj_in")
            or name.endswith("seed_proj_out")
            or name.endswith("seed_gate_vector")
            or name.endswith("seed_gate_bias")
            or name.endswith("z_lora_A")
            or name.endswith("z_lora_B")
            or name.endswith("out_lora_A")
            or name.endswith("out_lora_B")
        )
        if param.requires_grad:
            trainable.append(name)
    return trainable


def list_future_seed_parameters(model: nn.Module) -> list[str]:
    suffixes = (
        "fs_alpha",
        "delta_gain",
        "delta_bias",
        "seed_proj_in",
        "seed_proj_out",
        "seed_gate_vector",
        "seed_gate_bias",
        "z_lora_A",
        "z_lora_B",
        "out_lora_A",
        "out_lora_B",
    )
    return [name for name, _ in model.named_parameters() if name.endswith(suffixes)]


def get_future_seed_runtime_stats(model: nn.Module) -> dict[str, Any] | None:
    backbone = _get_text_backbone(model)
    return getattr(backbone, "_future_seed_last_runtime", None)
