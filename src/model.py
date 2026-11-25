"""
Kiến trúc Decoder-only Transformer hiện đại:
- RMSNorm
- RoPE (Rotary Position Embedding)
- Multi-Head Causal Self-Attention
- SwiGLU Feed-Forward Network
Bám sát tài liệu hướng dẫn công thức từng bước (LLM_CONG_THUC_TUNG_BUOC.md).
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from .config import ModelArgs
from gor_research.residuals import (
    BlockAttnResConnector,
    BlockAttnResGORv2Connector,
    DualBankGORv2Connector,
    StdDualBankConnector,
    _gor_v2_residual_add,
)


# -----------------------------------------------------------------------------
# 1. RMSNorm: Chuẩn hóa vector không cần trừ mean
# -----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS(x) = sqrt( mean(x^2) + eps )
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# -----------------------------------------------------------------------------
# 2. RoPE: Rotary Positional Embedding (Mã hóa vị trí quay)
# -----------------------------------------------------------------------------
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Tiền tính toán ma trận pha số phức e^(i * theta) cho các vị trí từ 0 đến end."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # e^(i * theta)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quay các vector Query và Key trong không gian 2D theo vị trí token."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[None, :xq_.shape[1], None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


# -----------------------------------------------------------------------------
# 3. Causal Multi-Head Self-Attention
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        
        # Ma trận chiếu tuyến tính cho Q, K, V và Output
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # Batch, Sequence Length, Channels (Embedding Dim)
        
        xq = self.wq(x)
        xk = self.wk(x)
        xv = self.wv(x)
        
        # Phân tách thành các heads: (B, T, n_heads, head_dim)
        xq = xq.view(B, T, self.n_heads, self.head_dim)
        xk = xk.view(B, T, self.n_heads, self.head_dim)
        xv = xv.view(B, T, self.n_heads, self.head_dim)
        
        # Áp dụng RoPE vào Query và Key
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # Chuyển về shape (B, n_heads, T, head_dim) để tính Attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Flash Attention / Scaled Dot-Product Attention có sẵn causal mask (tam giác dưới)
        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Nối các heads lại: (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(output)


# -----------------------------------------------------------------------------
# 4. SwiGLU Feed-Forward Network: W2 * (SiLU(W1 * x) * (W3 * x))
# -----------------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = None):
        super().__init__()
        if hidden_dim is None:
            # Chuẩn LLaMA: hidden_dim xấp xỉ 8/3 * dim và làm tròn bội số của 64
            hidden_dim = int(2 * (4 * dim) / 3)
            hidden_dim = (hidden_dim + 63) // 64 * 64
            
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# -----------------------------------------------------------------------------
# 4b. Orthogonal Sub-Expert Mixture-of-Experts (OS-MoE - Đột phá 2026)
# -----------------------------------------------------------------------------
class OrthogonalSubMoE(nn.Module):
    """
    Orthogonal Sub-Expert Mixture-of-Experts (OS-MoE):
    Kết hợp tinh hoa phân rã không gian con của Sub-MoE (AAAI 2026) và định tuyến trực giao.
    
    Cấu trúc:
    1. 1 Shared Expert (SwiGLU, dim=688): Luôn kích hoạt 100% token để bảo toàn ngữ pháp nền tảng.
    2. N Routed Sub-Experts (SwiGLU, dim=344): Phân bổ chuyên môn hóa sâu.
    3. Cosine-bounded Router: Triệt tiêu hiện tượng sụp đổ tầng sâu do chuẩn vector bị bùng nổ.
    4. Orthogonal Prototype Loss: Lực đẩy trực giao ép các chuyên gia vuông góc trong không gian Hilbert.
    """
    def __init__(self, dim: int, n_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts
        self.top_k = top_k
        
        # 1 Shared Expert: luôn kích hoạt (chiếm ~50% dung lượng FFN tiêu chuẩn)
        self.shared_expert = SwiGLU(dim, 688)
        
        # N Routed Sub-Experts: mỗi expert có kích thước gọn nhẹ
        self.sub_experts = nn.ModuleList([SwiGLU(dim, 344) for _ in range(n_experts)])
        
        # Router Gate với chuẩn hóa Cosine để chống sụp đổ ở tầng sâu
        self.gate = nn.Linear(dim, n_experts, bias=False)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        
        # 1. Nhánh Shared Expert
        shared_out = self.shared_expert(x)
        
        # 2. Định tuyến Cosine-bounded: chuẩn hóa cả x và vector prototype của gate
        norm_x = F.normalize(x_flat, p=2, dim=-1, eps=self.eps)
        norm_gate = F.normalize(self.gate.weight, p=2, dim=-1, eps=self.eps)
        router_logits = F.linear(norm_x, norm_gate) * 10.0  # Tỉ lệ nhiệt độ gamma = 10.0
        
        # 3. Chọn Top-K Sub-Experts
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)
        
        # 4. Gom kết quả từ các Sub-Experts bằng Scatter-Gather Vector hóa (Giải quyết triệt để Điểm nghẽn A)
        # Giảm số lần lặp từ K x E (16 lần) xuống E (8 lần) và tận dụng CUDA index_add_ nguyên tử
        routed_out = torch.zeros_like(x_flat)
        flat_topk_indices = top_k_indices.view(-1)
        flat_topk_weights = top_k_weights.view(-1, 1)
        token_indices = torch.arange(B * T, device=x.device).repeat_interleave(self.top_k)
        
        for e_idx in range(self.n_experts):
            expert_mask = (flat_topk_indices == e_idx)
            if expert_mask.any():
                t_idx = token_indices[expert_mask]
                w = flat_topk_weights[expert_mask]
                out_e = self.sub_experts[e_idx](x_flat[t_idx])
                routed_out.index_add_(0, t_idx, (out_e * w).to(routed_out.dtype))
                    
        total_out = shared_out + routed_out.view(B, T, C)
        
        # 5. Phạt phân rã trực giao (Orthogonal Prototype Repulsion Loss)
        # Ép các vector trọng số của Router vuông góc với nhau -> Chống sụp đổ theo bầy đàn
        gram = torch.mm(norm_gate, norm_gate.t())
        eye = torch.eye(self.n_experts, device=x.device)
        ortho_loss = ((gram - eye) ** 2).sum() / (self.n_experts * (self.n_experts - 1))
        
        return total_out, ortho_loss


# -----------------------------------------------------------------------------
# 5. Geometric Orthogonal Residual v2 (GOR-v2 - Đột phá kiến trúc 2026)
# -----------------------------------------------------------------------------
class GeometricOrthogonalResidual(nn.Module):
    """
    Geometric Orthogonal Residual v2 (GOR-v2):
    Cải tiến Residual Connection thuần túy bằng toán học hình học vi phân (0 tham số phụ).
    
    Phân rã bản cập nhật u = f(x) trong không gian Hilbert thành:
    1. u_perp (Thành phần trực giao - Tri thức mới hoàn toàn):
       Vuông góc với dòng chính x -> Luôn được cộng trọn vẹn 100%.
    2. u_parallel (Thành phần song song - Củng cố / Xung đột tri thức cũ):
       Điều biến bằng hàm mượt trơn C^infinity:
       phi(rho) = 2.0 * sigmoid(k * rho)
       - Tại rho = 0 (khởi tạo ngẫu nhiên): phi(0) = 1.0 -> Giữ nguyên 100% Identity Residual,
         loại bỏ hoàn toàn độ trễ khởi động ban đầu.
       - Khi rho > 0 (đồng thuận): Củng cố nhẹ tri thức tích lũy (lên tới ~1.5 - 2.0).
       - Khi rho < 0 (xung đột gradient): Dập tắt êm ả về sát 0, triệt tiêu nhiễu phá hoại.
    """
    def __init__(self, eps: float = 1e-7, k: float = 3.0):
        super().__init__()
        self.eps = eps
        self.k = k

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return _gor_v2_residual_add(x, u, k=self.k, eps=self.eps)


def _normalize_residual_type(residual_type: str) -> str:
    aliases = {
        "standard": "pre_norm",
        "gor": "gor_v2",
    }
    return aliases.get(residual_type, residual_type)


def _build_residual_connector(args: ModelArgs):
    mode = _normalize_residual_type(getattr(args, "residual_type", "gor"))
    num_sublayers = 2 * args.n_layers
    gor_k = getattr(args, "gor_k", 3.0)
    attn_num_blocks = getattr(args, "attn_num_blocks", 8)
    if mode in ("pre_norm", "gor_v2"):
        return None
    if mode == "std_dual_bank":
        return StdDualBankConnector(args.dim, num_sublayers=num_sublayers)
    if mode == "dual_bank_gor_v2":
        return DualBankGORv2Connector(args.dim, num_sublayers=num_sublayers, k=gor_k)
    if mode == "attn_res_block":
        return BlockAttnResConnector(args.dim, num_sublayers=num_sublayers, num_blocks=attn_num_blocks)
    if mode == "attn_res_gor_v2":
        return BlockAttnResGORv2Connector(
            args.dim, num_sublayers=num_sublayers, num_blocks=attn_num_blocks, k=gor_k
        )
    raise ValueError(f"Unsupported residual_type: {args.residual_type}")


# -----------------------------------------------------------------------------
# 6. Transformer Block (Hỗ trợ GOR-v2, Standard Residual, DualBank, AttnRes+GOR-v2, SwiGLU và OS-MoE)
# -----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.residual_mode = _normalize_residual_type(getattr(args, "residual_type", "gor"))
        self.ffn_type = getattr(args, "ffn_type", "swiglu")
        self.attention_norm = RMSNorm(args.dim)
        self.attention = CausalSelfAttention(args)
        self.ffn_norm = RMSNorm(args.dim)
        
        if self.ffn_type == "os_moe":
            self.feed_forward = OrthogonalSubMoE(
                dim=args.dim,
                n_experts=getattr(args, "n_routed_experts", 8),
                top_k=getattr(args, "num_experts_per_tok", 2)
            )
        else:
            self.feed_forward = SwiGLU(args.dim)

        if self.residual_mode == "gor_v2":
            gor_k = getattr(args, "gor_k", 3.0)
            self.res_attn = GeometricOrthogonalResidual(k=gor_k)
            self.res_ffn = GeometricOrthogonalResidual(k=gor_k)

    def forward_attention(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        return self.attention(self.attention_norm(x), freqs_cis)

    def forward_ffn(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm_h = self.ffn_norm(h)
        if self.ffn_type == "os_moe":
            return self.feed_forward(norm_h)
        return self.feed_forward(norm_h), torch.zeros((), device=h.device, dtype=h.dtype)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Nhánh 1: Multi-Head Attention
        u_attn = self.forward_attention(x, freqs_cis)
        h = self.res_attn(x, u_attn) if self.residual_mode == "gor_v2" else x + u_attn

        # Nhánh 2: Feed-Forward (SwiGLU hoặc OrthogonalSubMoE)
        u_ffn, aux_loss = self.forward_ffn(h)
        out = self.res_ffn(h, u_ffn) if self.residual_mode == "gor_v2" else h + u_ffn

        return out, aux_loss


# -----------------------------------------------------------------------------
# 7. MiniLLM: Mô hình ngôn ngữ hoàn chỉnh
# -----------------------------------------------------------------------------
class MiniLLM(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList([TransformerBlock(args) for _ in range(args.n_layers)])
        self.res_connector = _build_residual_connector(args)
        self.norm = RMSNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        
        # Weight tying: chia sẻ trọng số giữa bảng embedding và tầng output cuối
        self.output.weight = self.tok_embeddings.weight
        
        # Khởi tạo trọng số chuẩn mực LLaMA (std = 0.02) triệt tiêu hoàn toàn hiện tượng nổ loss ban đầu
        self.apply(self._init_weights)
        
        # Đăng ký ma trận tần số RoPE dạng buffer không cần học
        freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        h = self.tok_embeddings(idx)
        freqs_cis = self.freqs_cis[:T]
        
        total_aux_loss = torch.zeros((), device=idx.device, dtype=h.dtype)
        if self.res_connector is None:
            for layer in self.layers:
                h, layer_aux = layer(h, freqs_cis)
                total_aux_loss = total_aux_loss + layer_aux
        else:
            state = self.res_connector.init_state(h)
            sublayer_idx = 0
            for layer in self.layers:
                attn_in = self.res_connector.read(state, sublayer_idx)
                u_attn = layer.forward_attention(attn_in, freqs_cis)
                self.res_connector.update(state, u_attn, sublayer_idx)
                sublayer_idx += 1

                ffn_in = self.res_connector.read(state, sublayer_idx)
                u_ffn, layer_aux = layer.forward_ffn(ffn_in)
                self.res_connector.update(state, u_ffn, sublayer_idx)
                total_aux_loss = total_aux_loss + layer_aux
                sublayer_idx += 1
            h = self.res_connector.finalize(state)
            
        h = self.norm(h)
        logits = self.output(h)
        
        loss = None
        if targets is not None:
            # Cross-Entropy Loss: dự đoán token tiếp theo
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            moe_coeff = getattr(self.args, "moe_ortho_coeff", 0.01) if getattr(self.args, "ffn_type", "swiglu") == "os_moe" else 0.0
            loss = ce_loss + moe_coeff * total_aux_loss
            
        return logits, loss

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện trong mô hình."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_active_parameters(self) -> int:
        """Đếm số tham số được kích hoạt trên mỗi token (Active Parameters)."""
        if getattr(self.args, "ffn_type", "swiglu") != "os_moe":
            return self.count_parameters()
        
        # Với MoE: Active params = Toàn bộ mô hình - các inactive sub-experts
        n_routed = getattr(self.args, "n_routed_experts", 8)
        top_k = getattr(self.args, "num_experts_per_tok", 2)
        inactive_per_layer = (n_routed - top_k) * (3 * self.args.dim * 344)
        total_inactive = inactive_per_layer * self.args.n_layers
        return self.count_parameters() - total_inactive
