import inspect
import math
import types
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometricOrthogonalResidual(nn.Module):
    def __init__(self, eps: float = 1e-7, k: float = 3.0):
        super().__init__()
        self.eps = eps
        self.k = k

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        xf = x.float()
        uf = u.float()

        dot = (xf * uf).sum(dim=-1, keepdim=True)
        norm_sq_x = (xf * xf).sum(dim=-1, keepdim=True)
        norm_sq_u = (uf * uf).sum(dim=-1, keepdim=True)

        norm_x = torch.sqrt(norm_sq_x + self.eps)
        norm_u = torch.sqrt(norm_sq_u + self.eps)

        rho = dot / (norm_x * norm_u + self.eps)
        u_parallel = (dot / (norm_sq_x + self.eps)) * xf
        u_perp = uf - u_parallel

        phi = 2.0 * torch.sigmoid(self.k * rho)
        out = xf + u_perp + phi * u_parallel
        return out.to(dtype=orig_dtype)


def _gor_qwen2_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    position_embeddings=None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    attn_kwargs = {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "position_embeddings": position_embeddings,
    }
    attn_kwargs.update(kwargs)

    attn_outputs = self.self_attn(**attn_kwargs)
    if isinstance(attn_outputs, tuple):
        u_attn = attn_outputs[0]
        extra_attn = attn_outputs[1:]
    else:
        u_attn = attn_outputs
        extra_attn = ()

    hidden_states = self.gor_attn(residual, u_attn)

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    u_mlp = self.mlp(hidden_states)
    hidden_states = self.gor_mlp(residual, u_mlp)

    if getattr(self, "_gor_returns_tuple", False):
        outputs = (hidden_states,)
        if kwargs.get("output_attentions", False) and len(extra_attn) > 0:
            outputs += (extra_attn[0],)
        if use_cache and len(extra_attn) > 1:
            outputs += (extra_attn[1],)
        return outputs
    return hidden_states


def apply_gor_to_qwen_layers(model: nn.Module, k: float = 3.0) -> int:
    decoder_layers = _locate_decoder_layers(model)
    count = 0
    for layer in decoder_layers:
        if not hasattr(layer, "gor_attn"):
            layer.gor_attn = GeometricOrthogonalResidual(k=k)
        if not hasattr(layer, "gor_mlp"):
            layer.gor_mlp = GeometricOrthogonalResidual(k=k)
        sig = inspect.signature(type(layer).forward)
        ret_str = str(sig.return_annotation)
        layer._gor_returns_tuple = "tuple" in ret_str.lower() and "tensor" in ret_str.lower()
        layer.forward = types.MethodType(_gor_qwen2_decoder_layer_forward, layer)
        count += 1
    return count


class CosineBoundedRouter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        gamma: float = 10.0,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gamma = gamma
        self.gate = nn.Parameter(torch.empty(num_experts, hidden_dim, device=device, dtype=dtype))
        with torch.no_grad():
            init_w = torch.empty(num_experts, hidden_dim, device=device, dtype=torch.float32)
            nn.init.orthogonal_(init_w)
            self.gate.copy_(init_w.to(dtype=dtype if dtype is not None else torch.float32))

    def forward(self, x: torch.Tensor):
        xf = x.float()
        gf = self.gate.float()
        x_norm = F.normalize(xf, p=2, dim=-1, eps=1e-7)
        gate_norm = F.normalize(gf, p=2, dim=-1, eps=1e-7)
        logits = self.gamma * F.linear(x_norm, gate_norm)

        topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1).to(dtype=x.dtype)

        gram = torch.matmul(gate_norm, gate_norm.t())
        eye = torch.eye(self.num_experts, device=x.device, dtype=torch.float32)
        ortho_loss = torch.sum((gram - eye) ** 2) / (self.num_experts * (self.num_experts - 1))

        return topk_weights, topk_indices, logits, ortho_loss


class QwenSwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, device=None, dtype=None):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def _round_to_64(val: int) -> int:
    return max(64, int(round(val / 64.0)) * 64)


class QwenOSMoEBlock(nn.Module):
    def __init__(
        self,
        original_mlp: nn.Module,
        num_experts: int = 8,
        top_k: int = 2,
        moe_mode: str = "lite",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.moe_mode = moe_mode

        ref_w = original_mlp.gate_proj.weight
        device = ref_w.device
        dtype = ref_w.dtype

        hidden_size = original_mlp.gate_proj.in_features
        intermediate_size = original_mlp.gate_proj.out_features

        if moe_mode == "upcycled":
            shared_dim = _round_to_64(intermediate_size // 2)
            sub_dim = _round_to_64(intermediate_size // 4)
        else:
            shared_dim = _round_to_64(int(intermediate_size * 0.325))
            sub_dim = _round_to_64(int(intermediate_size * 0.093))

        self.shared_dim = shared_dim
        self.sub_dim = sub_dim
        self.hidden_size = hidden_size

        self.shared_expert = QwenSwiGLUExpert(hidden_size, shared_dim, device=device, dtype=dtype)
        self.router = CosineBoundedRouter(
            hidden_dim=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            gamma=10.0,
            device=device,
            dtype=dtype,
        )
        self.sub_experts = nn.ModuleList([
            QwenSwiGLUExpert(hidden_size, sub_dim, device=device, dtype=dtype)
            for _ in range(num_experts)
        ])

        self.last_ortho_loss = None
        self.last_router_logits = None
        self._initialize_from_dense_mlp(original_mlp, shared_dim, sub_dim)

    def _initialize_from_dense_mlp(self, original_mlp: nn.Module, shared_dim: int, sub_dim: int):
        with torch.no_grad():
            orig_gate = original_mlp.gate_proj.weight.data
            orig_up = original_mlp.up_proj.weight.data
            orig_down = original_mlp.down_proj.weight.data

            gate_n = orig_gate.float().norm(dim=1)
            up_n = orig_up.float().norm(dim=1)
            down_n = orig_down.float().norm(dim=0)
            importance = down_n * (gate_n + up_n)

            sorted_idx = torch.argsort(importance, descending=True)
            shared_idx = sorted_idx[:shared_dim]
            rem_idx = sorted_idx[shared_dim:]

            self.shared_expert.gate_proj.weight.copy_(orig_gate[shared_idx, :])
            self.shared_expert.up_proj.weight.copy_(orig_up[shared_idx, :])
            self.shared_expert.down_proj.weight.copy_(orig_down[:, shared_idx])

            pool = rem_idx if rem_idx.numel() >= sub_dim else sorted_idx
            pool_len = pool.numel()
            total_needed = self.num_experts * sub_dim

            if pool_len >= total_needed:
                full_pool = pool[:total_needed]
            else:
                repeats = (total_needed + pool_len - 1) // pool_len
                full_pool = pool.repeat(repeats)[:total_needed]

            down_scale = 1.0 / math.sqrt(self.top_k)
            for i, expert in enumerate(self.sub_experts):
                e_idx = full_pool[i * sub_dim : (i + 1) * sub_dim]
                noise_scale = 0.005 * (i + 1) / self.num_experts
                g_slice = orig_gate[e_idx, :]
                u_slice = orig_up[e_idx, :]
                d_slice = orig_down[:, e_idx] * down_scale

                expert.gate_proj.weight.copy_(g_slice + torch.randn_like(g_slice) * noise_scale)
                expert.up_proj.weight.copy_(u_slice + torch.randn_like(u_slice) * noise_scale)
                expert.down_proj.weight.copy_(d_slice + torch.randn_like(d_slice) * noise_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_out = self.shared_expert(x)
        topk_weights, topk_indices, raw_logits, ortho_loss = self.router(x)

        self.last_ortho_loss = ortho_loss
        self.last_router_logits = raw_logits.detach()

        B, S, D = x.shape
        x_flat = x.reshape(-1, D)
        routed_out = torch.zeros_like(x_flat)

        flat_indices = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1, 1)
        token_indices = torch.arange(B * S, device=x.device).repeat_interleave(self.top_k)

        for e_idx in range(self.num_experts):
            mask = flat_indices == e_idx
            if mask.any():
                t_idx = token_indices[mask]
                w = flat_weights[mask]
                out_e = self.sub_experts[e_idx](x_flat[t_idx])
                routed_out.index_add_(0, t_idx, (out_e * w).to(dtype=routed_out.dtype))

        return shared_out + routed_out.reshape(B, S, D)


def _locate_decoder_layers(model: nn.Module):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Unsupported model structure: unable to locate transformer layers.")


def build_qwen_variant(
    model: nn.Module,
    enable_gor: bool = True,
    enable_moe: bool = True,
    moe_mode: str = "lite",
    num_experts: int = 8,
    top_k: int = 2,
    gor_k: float = 3.0,
) -> nn.Module:
    decoder_layers = _locate_decoder_layers(model)

    if enable_moe:
        for idx in range(len(decoder_layers)):
            layer = decoder_layers[idx]
            if hasattr(layer, "mlp") and not isinstance(layer.mlp, QwenOSMoEBlock):
                layer.mlp = QwenOSMoEBlock(
                    original_mlp=layer.mlp,
                    num_experts=num_experts,
                    top_k=top_k,
                    moe_mode=moe_mode,
                )

    if enable_gor:
        apply_gor_to_qwen_layers(model, k=gor_k)

    return model


def upcycle_qwen_to_osmoe(
    model: nn.Module,
    num_experts: int = 8,
    top_k: int = 2,
    enable_gor: bool = True,
    moe_mode: str = "lite",
) -> nn.Module:
    return build_qwen_variant(
        model=model,
        enable_gor=enable_gor,
        enable_moe=True,
        moe_mode=moe_mode,
        num_experts=num_experts,
        top_k=top_k,
    )


def count_model_params(model: nn.Module) -> Dict[str, float]:
    total_params = sum(p.numel() for p in model.parameters())
    inactive_expert_params = 0
    for m in model.modules():
        if isinstance(m, QwenOSMoEBlock):
            per_expert = sum(p.numel() for p in m.sub_experts[0].parameters())
            inactive_expert_params += per_expert * (m.num_experts - m.top_k)
    active_params = total_params - inactive_expert_params
    return {
        "total_params": total_params,
        "active_params": active_params,
        "total_params_b": round(total_params / 1e9, 3),
        "active_params_b": round(active_params / 1e9, 3),
    }


def collect_ortho_loss(model: nn.Module, device) -> torch.Tensor:
    losses = []
    for m in model.modules():
        if isinstance(m, QwenOSMoEBlock) and m.last_ortho_loss is not None:
            losses.append(m.last_ortho_loss)
    if not losses:
        return torch.tensor(0.0, device=device)
    return torch.stack(losses).mean()
