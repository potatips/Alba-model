import inspect
import math
import types
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from qwen_moefication import _locate_decoder_layers


class RMSNormFP32(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * rms) * self.weight


def gor_v2_transform_u(
    x: torch.Tensor,
    u: torch.Tensor,
    k: float = 3.0,
    eps: float = 1e-7,
) -> torch.Tensor:
    xf = x.float()
    uf = u.float()
    dot = (xf * uf).sum(dim=-1, keepdim=True)
    norm_sq_x = (xf * xf).sum(dim=-1, keepdim=True) + eps
    norm_sq_u = (uf * uf).sum(dim=-1, keepdim=True) + eps
    rho = dot / torch.sqrt(norm_sq_x * norm_sq_u)
    phi = torch.clamp(2.0 * torch.sigmoid(k * rho), max=1.0)
    coeff = ((phi - 1.0) * dot) / norm_sq_x
    return (uf + coeff * xf).to(dtype=u.dtype)


class QwenBlockAttnResGraft(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        num_blocks: int = 9,
        enable_gor_v2: bool = False,
        gor_k: float = 3.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_sublayers = 2 * num_layers
        self.eff_blocks = max(1, min(num_blocks, self.num_sublayers))
        self.block_size = math.ceil(self.num_sublayers / self.eff_blocks)
        self.enable_gor_v2 = enable_gor_v2
        self.gor_k = gor_k

        self.alpha_attn = nn.Parameter(torch.zeros(num_layers, hidden_size, dtype=torch.float32))
        self.alpha_mlp = nn.Parameter(torch.zeros(num_layers, hidden_size, dtype=torch.float32))
        self.alpha_final = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))
        self.depth_queries = nn.Parameter(
            torch.zeros(self.num_sublayers + 1, hidden_size, dtype=torch.float32)
        )
        self.key_norms = nn.ModuleList(
            [RMSNormFP32(hidden_size) for _ in range(self.num_sublayers + 1)]
        )
        self.beta_gor = (
            nn.Parameter(torch.zeros(num_layers, dtype=torch.float32))
            if enable_gor_v2
            else None
        )

        self._completed: List[torch.Tensor] = []
        self._completed_keys: List[torch.Tensor] = []
        self._partial: Optional[torch.Tensor] = None

    def reset_bank(self, x0: torch.Tensor) -> None:
        self._completed = [x0]
        self._completed_keys = [self.key_norms[0](x0)]
        self._partial = None

    def _attend(
        self,
        cands: List[torch.Tensor],
        ckeys: List[torch.Tensor],
        query: torch.Tensor,
    ) -> torch.Tensor:
        if len(cands) == 1:
            return cands[0].float()
        logits = torch.stack([torch.einsum("d,...d->...", query, k) for k in ckeys], dim=0)
        weights = F.softmax(logits, dim=0)
        out = weights[0].unsqueeze(-1) * cands[0].float()
        for idx in range(1, len(cands)):
            out = out + weights[idx].unsqueeze(-1) * cands[idx].float()
        return out

    def read_blended_input(
        self,
        layer_idx: int,
        sublayer_offset: int,
        x_std: torch.Tensor,
    ) -> torch.Tensor:
        sublayer_idx = 2 * layer_idx + sublayer_offset
        if sublayer_idx == 0:
            return x_std
        if self._partial is None:
            cands = self._completed
            ckeys = self._completed_keys
        else:
            cands = self._completed + [self._partial]
            ckeys = self._completed_keys + [self.key_norms[sublayer_idx](self._partial)]
        h_bar = self._attend(cands, ckeys, self.depth_queries[sublayer_idx - 1])
        gate_param = self.alpha_attn[layer_idx] if sublayer_offset == 0 else self.alpha_mlp[layer_idx]
        gate = torch.tanh(gate_param)
        delta = (gate * (h_bar - x_std.float())).to(dtype=x_std.dtype)
        return x_std + delta

    def transform_attn_update(
        self,
        layer_idx: int,
        x_in: torch.Tensor,
        u_attn: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enable_gor_v2 or self.beta_gor is None:
            return u_attn
        u_gor = gor_v2_transform_u(x_in, u_attn, k=self.gor_k)
        bg = torch.tanh(self.beta_gor[layer_idx])
        delta = (bg * (u_gor.float() - u_attn.float())).to(dtype=u_attn.dtype)
        return u_attn + delta

    def push_update(self, sublayer_idx: int, u: torch.Tensor) -> None:
        self._partial = u if self._partial is None else (self._partial + u)
        is_end = ((sublayer_idx + 1) % self.block_size == 0) or (
            sublayer_idx + 1 == self.num_sublayers
        )
        if is_end:
            self._completed.append(self._partial)
            self._completed_keys.append(self.key_norms[sublayer_idx + 1](self._partial))
            self._partial = None

    def finalize_output(self, h_out: torch.Tensor) -> torch.Tensor:
        if len(self._completed) > 0:
            h_bar = self._attend(
                self._completed,
                self._completed_keys,
                self.depth_queries[self.num_sublayers],
            )
            gf = torch.tanh(self.alpha_final)
            delta = (gf * (h_bar - h_out.float())).to(dtype=h_out.dtype)
            res = h_out + delta
        else:
            res = h_out
        self._completed = []
        self._completed_keys = []
        self._partial = None
        return res

    def gate_stats(self) -> Dict[str, float]:
        with torch.no_grad():
            a_attn = torch.tanh(self.alpha_attn).abs()
            a_mlp = torch.tanh(self.alpha_mlp).abs()
            a_final = torch.tanh(self.alpha_final).abs()
            stats = {
                "mean_abs_alpha_attn": round(float(a_attn.mean().item()), 5),
                "max_abs_alpha_attn": round(float(a_attn.max().item()), 5),
                "mean_abs_alpha_mlp": round(float(a_mlp.mean().item()), 5),
                "mean_abs_alpha_final": round(float(a_final.mean().item()), 5),
            }
            if self.beta_gor is not None:
                b_gor = torch.tanh(self.beta_gor).abs()
                stats["mean_abs_beta_gor"] = round(float(b_gor.mean().item()), 5)
                stats["max_abs_beta_gor"] = round(float(b_gor.max().item()), 5)
            else:
                stats["mean_abs_beta_gor"] = 0.0
                stats["max_abs_beta_gor"] = 0.0
            return stats


def _grafted_qwen2_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=None,
    **kwargs,
):
    graft: QwenBlockAttnResGraft = self._graft_holder[0]
    layer_idx: int = self._graft_layer_idx
    num_layers: int = self._graft_num_layers

    if layer_idx == 0:
        graft.reset_bank(hidden_states)

    x_attn_in = graft.read_blended_input(layer_idx, 0, hidden_states)
    h_attn_norm = self.input_layernorm(x_attn_in)

    pkv = past_key_values if past_key_values is not None else past_key_value
    pkv_key = getattr(self, "_graft_attn_pkv_key", "past_key_values")
    attn_param_names = getattr(self, "_graft_attn_param_names", None)
    attn_accepts_varkw = getattr(self, "_graft_attn_accepts_varkw", True)

    candidate_kwargs = {
        "hidden_states": h_attn_norm,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        pkv_key: pkv,
        "use_cache": use_cache,
        "position_embeddings": position_embeddings,
    }
    candidate_kwargs.update(kwargs)

    if attn_param_names is not None and not attn_accepts_varkw:
        attn_kwargs = {k: v for k, v in candidate_kwargs.items() if k in attn_param_names}
    else:
        attn_kwargs = candidate_kwargs

    use_ckpt = self.training and getattr(self, "_graft_ckpt", False) and not use_cache

    if use_ckpt:
        static_attn_kwargs = {k: v for k, v in attn_kwargs.items() if k != "hidden_states"}

        def _attn_fn(h: torch.Tensor):
            out = self.self_attn(hidden_states=h, **static_attn_kwargs)
            return out[0] if isinstance(out, tuple) else out

        u_attn = torch_checkpoint(_attn_fn, h_attn_norm, use_reentrant=False)
        extra_attn: Tuple = ()
    else:
        attn_outputs = self.self_attn(**attn_kwargs)
        if isinstance(attn_outputs, tuple):
            u_attn = attn_outputs[0]
            extra_attn = attn_outputs[1:]
        else:
            u_attn = attn_outputs
            extra_attn = ()

    u_attn_eff = graft.transform_attn_update(layer_idx, x_attn_in, u_attn)
    h_mid = x_attn_in + u_attn_eff
    graft.push_update(2 * layer_idx, u_attn_eff)

    x_mlp_in = graft.read_blended_input(layer_idx, 1, h_mid)
    h_mlp_norm = self.post_attention_layernorm(x_mlp_in)
    if use_ckpt:
        u_mlp = torch_checkpoint(self.mlp, h_mlp_norm, use_reentrant=False)
    else:
        u_mlp = self.mlp(h_mlp_norm)

    h_out = x_mlp_in + u_mlp
    graft.push_update(2 * layer_idx + 1, u_mlp)

    if layer_idx == num_layers - 1:
        h_out = graft.finalize_output(h_out)

    if getattr(self, "_graft_returns_tuple", False):
        outputs = (h_out,)
        if kwargs.get("output_attentions", False) and len(extra_attn) > 0:
            outputs += (extra_attn[0],)
        if use_cache and len(extra_attn) > 1:
            outputs += (extra_attn[1],)
        return outputs
    return h_out


def apply_residual_graft_to_qwen(
    model: nn.Module,
    enable_gor_v2: bool = False,
    num_blocks: int = 9,
    gor_k: float = 3.0,
    use_sublayer_checkpointing: bool = True,
) -> QwenBlockAttnResGraft:
    decoder_layers = _locate_decoder_layers(model)
    num_layers = len(decoder_layers)
    hidden_size = getattr(getattr(model, "config", None), "hidden_size", None)
    if hidden_size is None:
        hidden_size = decoder_layers[0].input_layernorm.weight.numel()

    ref_param = next(model.parameters())
    graft = QwenBlockAttnResGraft(
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_blocks=num_blocks,
        enable_gor_v2=enable_gor_v2,
        gor_k=gor_k,
    ).to(device=ref_param.device, dtype=torch.float32)

    model.residual_graft = graft

    for idx, layer in enumerate(decoder_layers):
        layer._graft_holder = (graft,)
        layer._graft_layer_idx = idx
        layer._graft_num_layers = num_layers
        layer._graft_ckpt = use_sublayer_checkpointing

        sig = inspect.signature(type(layer).forward)
        ret_str = str(sig.return_annotation)
        layer._graft_returns_tuple = "tuple" in ret_str.lower() and "tensor" in ret_str.lower()

        attn_sig = inspect.signature(layer.self_attn.forward)
        attn_params = attn_sig.parameters
        layer._graft_attn_param_names = set(attn_params.keys())
        layer._graft_attn_accepts_varkw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in attn_params.values()
        )
        layer._graft_attn_pkv_key = (
            "past_key_values" if "past_key_values" in attn_params else "past_key_value"
        )
        layer.forward = types.MethodType(_grafted_qwen2_decoder_layer_forward, layer)

    return graft


def collect_graft_gate_stats(model: nn.Module) -> Dict[str, float]:
    graft = getattr(model, "residual_graft", None)
    if isinstance(graft, QwenBlockAttnResGraft):
        return graft.gate_stats()
    return {
        "mean_abs_alpha_attn": 0.0,
        "max_abs_alpha_attn": 0.0,
        "mean_abs_alpha_mlp": 0.0,
        "mean_abs_alpha_final": 0.0,
        "mean_abs_beta_gor": 0.0,
        "max_abs_beta_gor": 0.0,
    }
