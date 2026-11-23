import math
from typing import Tuple
import torch
from torch.optim.optimizer import Optimizer


def zeroth_power_via_newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() if G.dtype == torch.bfloat16 else G.float()
    if X.size(0) > X.size(1):
        X = X.t()
        transposed = True
    else:
        transposed = False

    X = X / (X.norm() + eps)

    for _ in range(steps):
        A = torch.mm(X, X.t())
        B = b * A + c * torch.mm(A, A)
        X = a * X + torch.mm(B, X)

    if transposed:
        X = X.t()

    return X.to(dtype=G.dtype)


class GEOMuon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            ns_steps=ns_steps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            is_muon = group.get("is_muon", False)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    if is_muon and p.dim() >= 2:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    else:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step = state["step"]

                if is_muon and p.dim() >= 2:
                    if weight_decay != 0:
                        p.mul_(1.0 - lr * weight_decay)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)
                    nesterov_g = grad.add(buf, alpha=momentum)

                    g_2d = nesterov_g.reshape(nesterov_g.size(0), -1)
                    ortho_update = zeroth_power_via_newton_schulz_5(g_2d, steps=ns_steps)
                    ortho_update = ortho_update.reshape_as(p)

                    d_out, d_in = g_2d.size(0), g_2d.size(1)
                    rms_scale = 0.2 * math.sqrt(max(d_out, d_in))

                    p.add_(ortho_update, alpha=-lr * rms_scale)
                else:
                    if weight_decay != 0:
                        p.mul_(1.0 - lr * weight_decay)

                    g_f = grad.float()
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(g_f, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_f, g_f, value=1.0 - beta2)

                    bc1 = 1.0 - (beta1 ** step)
                    bc2 = 1.0 - (beta2 ** step)

                    step_size = lr / bc1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bc2)).add_(eps)
                    update = (exp_avg / denom).to(dtype=p.dtype)
                    p.add_(update, alpha=-step_size)

        return loss


def create_qwen_geomuon_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    ns_steps: int = 5,
) -> GEOMuon:
    muon_params = []
    adamw_params = []
    adamw_no_decay = []

    target_2d_keys = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2 and any(k in name for k in target_2d_keys):
            muon_params.append(param)
        elif param.dim() < 2 or "norm" in name or "bias" in name:
            adamw_no_decay.append(param)
        else:
            adamw_params.append(param)

    param_groups = [
        {"params": muon_params, "is_muon": True, "lr": lr, "weight_decay": weight_decay},
        {"params": adamw_params, "is_muon": False, "lr": lr * 0.5, "weight_decay": weight_decay},
        {"params": adamw_no_decay, "is_muon": False, "lr": lr * 0.5, "weight_decay": 0.0},
    ]

    return GEOMuon(
        param_groups,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        ns_steps=ns_steps,
    )


def create_qwen_adamw_optimizer(
    model: torch.nn.Module,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    use_fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    except Exception:
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)
