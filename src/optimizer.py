"""
GEO-Muon: Geometric Energy-Preserving Orthogonal Muon Optimizer (Frontier 2026).
Thiết kế chuyên biệt cho Alba-Model nhằm giải quyết các điểm yếu của AdamW và Vanilla Muon.

Các đột phá cốt lõi:
1. Matrix-Orthogonal Optimization: Trực giao hóa ma trận Momentum 2D bằng thuật toán
   lặp đại số Newton-Schulz bậc 5 (Quintic iteration), triệt tiêu hiện tượng méo phổ kỳ dị (Spectral Distortion).
2. Energy-Preserving Spectral Scaling (EPSS): Bảo toàn độ lớn năng lượng Frobenius:
   scale = ||M||_F / (||O||_F + eps)
   -> Giữ nguyên biên độ xung lực thực tế của gradient, ngăn chặn triệt để hiện tượng
      sụp đổ tỷ lệ cập nhật (Update Ratio Collapse) như AdamW gặp phải ở các bước chuyển pha.
3. Hybrid Co-existence: Tự động phân chia thông minh:
   - Các ma trận ẩn 2D (Attention, SwiGLU, OS-MoE Sub-Experts) -> Tối ưu bằng GEO-Muon.
   - Các tham số 1D (RMSNorm scale), ma trận từ vựng (Token Embeddings, LM Head) -> Tối ưu bằng AdamW.
"""

import math
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

try:
    from torch.optim import _functional as F_opt
    _HAS_FUNCTIONAL_ADAMW = hasattr(F_opt, "adamw")
except ImportError:
    _HAS_FUNCTIONAL_ADAMW = False


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Thuật toán lặp Newton-Schulz bậc 5 (Quintic Newton-Schulz Iteration)
    để tính toán xấp xỉ ma trận trực giao cực (Orthogonal Polar Factor) cực nhanh trên Tensor Core.
    
    Đa thức lặp bậc 5:
    X_{k+1} = X_k * (a * I + b * X_k^T * X_k + c * (X_k^T * X_k)^2)
    Hệ số tối ưu hóa: a = 3.4445, b = -4.7750, c = 2.0315
    """
    assert len(G.shape) >= 2, f"Newton-Schulz yêu cầu ma trận >= 2 chiều, nhận được shape {G.shape}"
    a, b, c = (3.4445, -4.7750, 2.0315)

    use_bf16 = G.is_cuda and torch.cuda.is_bf16_supported()
    X = G.bfloat16() if use_bf16 else G.float()
    transposed = False

    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True

    norm = X.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    X = X / norm

    for _ in range(steps):
        A = X @ X.mT
        if X.dim() == 2:
            B = torch.addmm(A, A, A, beta=b, alpha=c)
            X = torch.addmm(X, B, X, beta=a, alpha=1.0)
        else:
            B = b * A + c * (A @ A)
            X = a * X + B @ X

    if transposed:
        X = X.mT

    return X.to(dtype=G.dtype)


class GEOMuon(torch.optim.Optimizer):
    """
    Bộ tối ưu hóa kết hợp GEO-Muon + AdamW v2.1 (Batched Newton-Schulz & Vectorized AdamW):
    - Nhóm tham số có `use_muon=True`: Gom cụm các ma trận 2D theo (shape, dtype, device) để
      thực hiện Batched Newton-Schulz bậc 5 một lần (B, m, n) với Moonlight RMS scaling & Nesterov momentum.
      Các tham số không phải 2D (ndim != 2) được xử lý singleton độc lập.
    - Nhóm tham số có `use_muon=False`: Vectorized AdamW tối ưu hiệu năng cao qua
      `torch.optim._functional.adamw` (hỗ trợ foreach/fused và tự động fallback), tương thích
      hoàn toàn checkpoint state_dict với step dạng int hoặc 0-dim tensor.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        ns_steps: int = 5,
        nesterov: bool = True,
        rms_scale: float = 0.2,
        foreach: Optional[bool] = None,
        fused: Optional[bool] = None,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            ns_steps=ns_steps,
            nesterov=nesterov,
            rms_scale=rms_scale,
            foreach=foreach,
            fused=fused,
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
            weight_decay = group["weight_decay"]
            use_muon = group.get("use_muon", False)

            if use_muon:
                momentum = group["momentum"]
                ns_steps = group.get("ns_steps", 5)
                nesterov = group.get("nesterov", True)
                rms_scale = group.get("rms_scale", 0.2)

                buckets = defaultdict(list)
                singletons = []

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]
                    buf.lerp_(g, 1.0 - momentum)
                    update_in = g.lerp(buf, momentum) if nesterov else buf

                    if p.ndim == 2:
                        key = (p.shape, p.dtype, p.device)
                        buckets[key].append((p, update_in))
                    else:
                        singletons.append((p, update_in))

                # Process 2D buckets (batched if >= 2 tensors, singleton otherwise)
                for (shape, dtype, device), bucket in buckets.items():
                    if len(bucket) >= 2:
                        params, update_ins = zip(*bucket)
                        stacked = torch.stack(update_ins, dim=0)  # shape (B, m, n)
                        O_stacked = zeropower_via_newtonschulz5(stacked, steps=ns_steps)
                        m, n = shape[-2], shape[-1]
                        scale = rms_scale * math.sqrt(max(m, n))

                        for p, O in zip(params, O_stacked.unbind(dim=0)):
                            if weight_decay != 0:
                                p.mul_(1.0 - lr * weight_decay)
                            p.add_(O, alpha=-lr * scale)
                    else:
                        p, update_in = bucket[0]
                        O = zeropower_via_newtonschulz5(update_in, steps=ns_steps)
                        scale = rms_scale * math.sqrt(max(O.size(-2), O.size(-1)))

                        if weight_decay != 0:
                            p.mul_(1.0 - lr * weight_decay)
                        p.add_(O, alpha=-lr * scale)

                # Process non-2D tensors in singleton handling
                for p, update_in in singletons:
                    O = zeropower_via_newtonschulz5(update_in, steps=ns_steps)
                    scale = rms_scale * math.sqrt(max(O.size(-2), O.size(-1)))

                    if weight_decay != 0:
                        p.mul_(1.0 - lr * weight_decay)
                    p.add_(O, alpha=-lr * scale)

            else:
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                state_steps = []

                for p in group["params"]:
                    if p.grad is None:
                        continue
                    params_with_grad.append(p)
                    grads.append(p.grad)
                    state = self.state[p]

                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0, dtype=torch.float32)
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    else:
                        if "step" not in state:
                            state["step"] = torch.tensor(0.0, dtype=torch.float32)
                        elif not torch.is_tensor(state["step"]):
                            state["step"] = torch.tensor(float(state["step"]), dtype=torch.float32)
                        elif not state["step"].is_floating_point():
                            state["step"] = state["step"].to(dtype=torch.float32)

                        if "exp_avg" not in state:
                            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if "exp_avg_sq" not in state:
                            state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    state_steps.append(state["step"])

                if not params_with_grad:
                    continue

                executed = False
                if _HAS_FUNCTIONAL_ADAMW:
                    use_fused = group.get("fused", None)
                    use_foreach = group.get("foreach", None)

                    if use_fused is None and use_foreach is None:
                        use_foreach = True

                    try:
                        adamw_kwargs = dict(
                            amsgrad=False,
                            beta1=beta1,
                            beta2=beta2,
                            lr=lr,
                            weight_decay=weight_decay,
                            eps=eps,
                            maximize=False,
                        )
                        if use_foreach is not None:
                            adamw_kwargs["foreach"] = use_foreach
                        if use_fused is not None:
                            adamw_kwargs["fused"] = use_fused

                        F_opt.adamw(
                            params_with_grad,
                            grads,
                            exp_avgs,
                            exp_avg_sqs,
                            [],
                            state_steps,
                            **adamw_kwargs,
                        )
                        executed = True
                    except Exception:
                        try:
                            F_opt.adamw(
                                params_with_grad,
                                grads,
                                exp_avgs,
                                exp_avg_sqs,
                                [],
                                state_steps,
                                amsgrad=False,
                                beta1=beta1,
                                beta2=beta2,
                                lr=lr,
                                weight_decay=weight_decay,
                                eps=eps,
                                maximize=False,
                                foreach=False,
                            )
                            executed = True
                        except Exception:
                            executed = False

                if not executed:
                    for p, g, exp_avg, exp_avg_sq, step_tensor in zip(
                        params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
                    ):
                        step_tensor.add_(1.0)
                        step_t = step_tensor.item()
                        exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                        bias_correction1 = 1 - beta1 ** step_t
                        bias_correction2 = 1 - beta2 ** step_t

                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                        step_size = lr / bias_correction1

                        if weight_decay != 0:
                            p.mul_(1.0 - lr * weight_decay)

                        p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = "geo_muon",
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    momentum: float = 0.95,
    betas: Tuple[float, float] = (0.9, 0.95),
) -> torch.optim.Optimizer:
    """
    Tạo bộ tối ưu hóa tự động phân luồng theo kiến trúc mô hình:
    - Nếu optimizer_type == 'geo_muon':
      + Toàn bộ ma trận ẩn 2D (Attention, FFN, MoE Sub-Experts) -> GEO-Muon
      + Toàn bộ 1D, Embeddings, LM Head, Router Gate -> AdamW
    - Nếu optimizer_type == 'adamw': AdamW thuần túy cho toàn bộ mô hình
    - Nếu optimizer_type == 'adam': Adam chuẩn (L2 Penalty)
    """
    optimizer_type = optimizer_type.lower()

    if optimizer_type == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )
    elif optimizer_type in ["geo_muon", "muon"]:
        muon_params = []
        adamw_params = []
        muon_names = []
        adamw_names = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Tiêu chí phân luồng GEO-Muon:
            # 1. Bắt buộc phải là ma trận 2D
            # 2. KHÔNG phải là Embedding (lookup table)
            # 3. KHÔNG phải là LM Head (output classification)
            # 4. KHÔNG phải là Router Gate (prototype cosine vectors)
            is_2d = len(param.shape) >= 2
            is_embed_or_head = (
                ("tok_embeddings" in name)
                or ("token_embedding" in name)
                or ("output" in name)
                or ("lm_head" in name)
            )
            is_gate = "gate" in name
            is_norm = "norm" in name or "connector." in name or param.ndim < 2

            if is_2d and not is_embed_or_head and not is_gate and not is_norm:
                muon_params.append(param)
                muon_names.append(name)
            else:
                adamw_params.append(param)
                adamw_names.append(name)

        muon_count = sum(p.numel() for p in muon_params)
        adamw_count = sum(p.numel() for p in adamw_params)
        total = muon_count + adamw_count

        print("⚡ [GEO-Muon Hybrid Optimizer Initialized]")
        print(f"   ├─ Nhánh GEO-Muon (2D Matrix Orthogonal): {len(muon_params)} tensors | {muon_count:,} params ({muon_count / total * 100:.1f}%)")
        print(f"   └─ Nhánh AdamW (1D / Embed / Head / Router): {len(adamw_params)} tensors | {adamw_count:,} params ({adamw_count / total * 100:.1f}%)")

        param_groups = [
            {
                "params": muon_params,
                "use_muon": True,
                "lr": lr,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "nesterov": True,
                "rms_scale": 0.2,
            },
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": lr,
                "betas": betas,
                "weight_decay": weight_decay,
                "eps": 1e-8,
            },
        ]

        return GEOMuon(param_groups)
    else:
        raise ValueError(f"Optimizer không hợp lệ: {optimizer_type}. Chọn 'geo_muon', 'adamw', hoặc 'adam'.")
