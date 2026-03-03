"""
DynFormer: Dynamics-Informed Transformer Neural Operator for PDEs.

Architectural Hierarchy
-----------------------
1. DynamicsEmbedding       — FFT / physical-domain feature extraction (ablation: spectral / physical, scale decomposition)
2. KroneckerAttention      — Factored global attention via Kronecker structure
3. LinearAttentionBlock    — O(N) linear attention (ablation alternative)
4. ClassicalAttentionBlock — O(N²) softmax attention (ablation alternative)
5. LGM_Transformation     — Local-Global-Mixing with configurable interaction (mixing / adding / global_only / local_only)
6. FullScaleDynamicsLayer  — Single FSDL block with linear + nonlinear branches
7. DynFormer               — Top-level model: Lifting → L×FSDL → Projection

Ablation Properties
-------------------
- global_kernel: 'KSAttention' | 'linear_attention' | 'classical_attention'
- embedding_type: 'spectral' | 'physical'
- use_scale_decomposition: True | False
- lgm_type: 'mixing' | 'adding' | 'global_only' | 'local_only'
- architecture_type: 'hybrid' | 'only_sequential' | 'only_parallel'
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


# =============================================================================
# Utility Modules
# =============================================================================


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable depth, activation, and dropout.

    Parameters
    ----------
    dims : List[int]
        Dimensions for each layer.
    act_fn : nn.Module
        Activation function applied between hidden layers.
    dropout : float, optional
        Dropout probability. Default: 0.0.
    """

    def __init__(self, dims: List[int], act_fn: nn.Module, dropout: float = 0.0):
        super().__init__()
        layers: list = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """``[..., dims[0]] → [..., dims[-1]]``."""
        return self.net(x)


# =============================================================================
# Positional Embedding Utilities
# =============================================================================


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE)."""

    def __init__(self, dim: int, min_freq: float = 1 / 64, scale: float = 1.0):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coordinates: Tensor, device: torch.device) -> Tensor:
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum("... i , j -> ... i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t: Tensor, freqs_x: Tensor, freqs_y: Tensor) -> Tensor:
    d = t.shape[-1]
    t_x, t_y = t[..., : d // 2], t[..., d // 2 :]
    return torch.cat([apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)], dim=-1)


# =============================================================================
# Core Attention Components
# =============================================================================


class _PositionAwareReductionKernel(nn.Module):
    r"""Low-rank kernel producing :math:`\mathcal{K} \in \mathbb{R}^{N \times N}`
    for one coordinate axis using learned Q/K projections + optional RoPE.

    Parameters
    ----------
    dim, dim_head, heads, use_rope, scaling : see KroneckerAttention.
    """

    def __init__(self, dim: int, dim_head: int, heads: int,
                 use_rope: bool = True, scaling: float = 1.0):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        self.scaling = scaling
        self.ffn_q = MLP([dim, dim_head * heads], nn.GELU())
        self.ffn_k = MLP([dim, dim_head * heads], nn.GELU())
        self.pos_emb: Optional[RotaryEmbedding] = None
        if use_rope:
            self.pos_emb = RotaryEmbedding(dim_head, min_freq=1 / 64)

    def forward(self, u: Tensor, pos: Optional[Tensor] = None) -> Tensor:
        """``[B, N, C] → [B, heads, N, N]``."""
        q = rearrange(self.ffn_q(u), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.ffn_k(u), "b n (h d) -> b h n d", h=self.heads)
        if self.pos_emb is not None and pos is not None:
            freqs = self.pos_emb(pos[..., 0], q.device).unsqueeze(0)
            freqs = repeat(freqs, "1 n d -> b h n d", b=q.shape[0], h=self.heads)
            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)
        return torch.einsum("bhid,bhjd->bhij", q, k) * self.scaling


class DynamicsEmbedding(nn.Module):
    r"""Dynamics Embedding with configurable embedding type and scale decomposition.

    Embeds the input via either spectral (FFT-based) or physical-domain
    processing.  Returns a single embedded tensor ``[B, H, W, C]``.
    Downstream projection into Q/K/V is handled by the attention block.

    Parameters
    ----------
    dim : int
        Input channel dimension.
    modes : Tuple[int, int]
        Fourier modes ``(M1, M2)``.
    embedding_type : str
        ``"spectral"`` or ``"physical"``. Default: ``"spectral"``.
    use_scale_decomposition : bool
        If True, truncate to ``(M1, M2)`` modes; if False, use all modes.
        Default: True.
    """

    def __init__(
        self,
        dim: int,
        modes: Tuple[int, int],
        embedding_type: str = "spectral",
        use_scale_decomposition: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.modes1, self.modes2 = modes
        self.embedding_type = embedding_type
        self.use_scale_decomposition = use_scale_decomposition

        if embedding_type == "spectral":
            scale = 1.0 / (dim * dim)
            self.spectral_weights_pos = nn.Parameter(
                scale * torch.randn(dim, dim, self.modes1, self.modes2, dtype=torch.cfloat))
            self.spectral_weights_neg = nn.Parameter(
                scale * torch.randn(dim, dim, self.modes1, self.modes2, dtype=torch.cfloat))
        elif embedding_type == "physical":
            self.proj_all = nn.Linear(dim, dim, bias=False)
        else:
            raise ValueError(f"Unknown embedding_type: {embedding_type}")

    @staticmethod
    def _compl_mul2d(inp: Tensor, weights: Tensor) -> Tensor:
        return torch.einsum("bxyi,ioxy->bxyo", inp, weights)

    def forward(self, u: Tensor) -> Tensor:
        """
        Parameters
        ----------
        u : Tensor
            ``[B, H, W, C]``.

        Returns
        -------
        Tensor
            Embedded features ``[B, H, W, C]``.
        """
        B, H, W, C = u.shape
        u_ft = torch.fft.rfft2(u, dim=(-3, -2))

        if self.embedding_type == "spectral":
            if self.use_scale_decomposition:
                out_ft = torch.zeros(B, H, W, C, dtype=torch.cfloat, device=u.device)
                out_ft[:, :self.modes1, :self.modes2, :] = self._compl_mul2d(
                    u_ft[:, :self.modes1, :self.modes2, :], self.spectral_weights_pos)
                out_ft[:, -self.modes1:, :self.modes2, :] = self._compl_mul2d(
                    u_ft[:, -self.modes1:, :self.modes2, :], self.spectral_weights_neg)
                out_ft[:, -self.modes1:self.modes1, :self.modes2, :] = \
                    u_ft[:, -self.modes1:self.modes1, :self.modes2, :].clone()
                u = torch.fft.irfft2(out_ft, s=(H, W), dim=(-3, -2))
            else:
                m1 = min(self.modes1, H)
                m2 = min(self.modes2, W // 2 + 1)
                out_ft = torch.zeros_like(u_ft)
                out_ft[:, :m1, :m2, :] = self._compl_mul2d(
                    u_ft[:, :m1, :m2, :], self.spectral_weights_pos[:, :, :m1, :m2])
                out_ft[:, -m1:, :m2, :] = self._compl_mul2d(
                    u_ft[:, -m1:, :m2, :], self.spectral_weights_neg[:, :, :m1, :m2])
                out_ft[:, m1:-m1, :, :] = u_ft[:, m1:-m1, :, :].clone()
                u = torch.fft.irfft2(out_ft, s=(H, W), dim=(-3, -2))
        else:
            m1 = min(self.modes1, H)
            m2 = min(self.modes2, W // 2 + 1)
            m1_down, m1_up = m1 // 2, m1 - m1 // 2
            u_ft_low = torch.zeros(B, m1, m2, C, dtype=torch.cfloat, device=u_ft.device)
            u_ft_low[:, :m1_down, :m2, :] = u_ft[:, :m1_down, :m2, :]
            u_ft_low[:, -m1_up:, :m2, :] = u_ft[:, -m1_up:, :m2, :]
            u_low = torch.fft.irfft2(u_ft_low, s=(m1, 2 * (m2 - 1)), dim=(-3, -2))
            u = self.proj_all(u_low)
            if u.shape[1] != H or u.shape[2] != W:
                u = u.permute(0, 3, 1, 2)
                u = F.interpolate(u, size=(H, W), mode="bilinear", align_corners=False)
                u = u.permute(0, 2, 3, 1)

        return u


class KroneckerAttention(nn.Module):
    r"""Kronecker-Factored Attention: :math:`\mathcal{K}_1 \tilde{V} \mathcal{K}_2^\top`.

    Accepts an embedded tensor ``[B, H, W, C]`` and internally projects it
    into per-axis reduction features (for kernel matrices) and multi-head
    value features before applying the Kronecker-factored attention.

    Parameters
    ----------
    dim : int
        Input channel dimension.
    dim_head : int
        Dimension per attention head (for V).
    heads : int
        Number of attention heads.
    dim_out : int
        Output channel dimension.
    kernel_multiplier : int
        Multiplier for Q/K head dim relative to V. Default: 3.
    scaling_factor : float
        Default: 1.0.
    use_rope : bool
        Default: True.
    """

    def __init__(self, dim: int, dim_head: int, heads: int, dim_out: int,
                 kernel_multiplier: int = 3, scaling_factor: float = 1.0,
                 use_rope: bool = True):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        qk_dim_head = dim_head * kernel_multiplier

        # Feature projection (from embedding)
        self.norm = nn.LayerNorm(dim)
        self.to_features = nn.Linear(dim, dim, bias=False)

        # Per-axis reduction projections (for Q/K generation)
        self.to_axis_x = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False))
        self.to_axis_y = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False), nn.Linear(dim, dim, bias=False))

        # Value projection
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)

        # Reduction kernels for each spatial axis
        self.kernel_x = _PositionAwareReductionKernel(
            dim, qk_dim_head, heads, use_rope=use_rope, scaling=scaling_factor)
        self.kernel_y = _PositionAwareReductionKernel(
            dim, qk_dim_head, heads, use_rope=use_rope, scaling=scaling_factor)

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim_head * heads),
            nn.Linear(dim_head * heads, dim_out, bias=False))

    def forward(self, u: Tensor) -> Tensor:
        """``[B, H, W, C] → [B, H, W, dim_out]``."""
        # Project embedding into features
        u = self.norm(u)
        u = self.to_features(u)

        v = self.to_v(u)                      # [B, H, W, heads*dim_head]
        u_x = self.to_axis_x(u.mean(dim=2))   # [B, H, C]
        u_y = self.to_axis_y(u.mean(dim=1))   # [B, W, C]

        H_s, W_s = u_x.shape[1], u_y.shape[1]
        pos_x = torch.linspace(0, 2 * math.pi, H_s, device=u_x.device).unsqueeze(-1)
        pos_y = torch.linspace(0, 2 * math.pi, W_s, device=u_y.device).unsqueeze(-1)
        k_1 = self.kernel_x(u_x, pos=pos_x)
        k_2 = self.kernel_y(u_y, pos=pos_y)
        v_mh = rearrange(v, "b i l (h c) -> b h i l c", h=self.heads)
        out = torch.einsum("bhij,bhjmc->bhimc", k_1, v_mh)
        out = torch.einsum("bhlm,bhimc->bhilc", k_2, out)
        out = rearrange(out, "b h i l c -> b i l (h c)")
        return self.to_out(out)


# =============================================================================
# Alternative Attention Mechanisms (for ablation)
# =============================================================================


class LinearAttentionBlock(nn.Module):
    """Linear-complexity attention (Fourier-type).

    Replaces KroneckerAttention for ablation experiment E1.
    Input/output: ``[B, H, W, C] → [B, H, W, C]``.

    Parameters
    ----------
    dim : int
        Channel dimension.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per head.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.q_norm = nn.LayerNorm(dim_head)
        self.k_norm = nn.LayerNorm(dim_head)

    def _norm_wrt_domain(self, x: Tensor, norm_fn: nn.Module) -> Tensor:
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, "b h n d -> (b h) n d")),
            "(b h) n d -> b h n d", b=b)

    def forward(self, x: Tensor) -> Tensor:
        """``[B, H, W, C] → [B, H, W, C]``."""
        _, l, w, _ = x.shape
        x = rearrange(x, "b l w c -> b (l w) c")
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        q = self._norm_wrt_domain(q, self.q_norm)
        k = self._norm_wrt_domain(k, self.k_norm)
        dots = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, dots) * (1.0 / q.shape[2])
        return rearrange(out, "b h (l w) d -> b l w (h d)", l=l, w=w)


class ClassicalAttentionBlock(nn.Module):
    """Classical O(N²) softmax attention.

    Replaces KroneckerAttention for ablation experiment E1.
    Input/output: ``[B, H, W, C] → [B, H, W, C]``.

    Parameters
    ----------
    dim : int
        Channel dimension.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per head.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        """``[B, H, W, C] → [B, H, W, C]``."""
        B, H, W, C = x.shape
        x = rearrange(x, "b h w c -> b (h w) c")
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)
        attn = (torch.matmul(q, k.transpose(-1, -2)) * self.scale).softmax(dim=-1)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return rearrange(self.to_out(out), "b (h w) c -> b h w c", h=H, w=W)


# =============================================================================
# Layer Modules
# =============================================================================


class LGM_Transformation(nn.Module):
    r"""Local-Global-Mixing (LGM) Transformation with configurable interaction.

    Parameters
    ----------
    dim : int
        Channel dimension.
    modes : Tuple[int, int]
        Fourier modes ``(M1, M2)``.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per head.
    local_kernel : str
        ``"mlp"`` or ``"conv"``.
    kernel_size : int
        Conv kernel size.
    lgm_type : str
        Interaction mode: ``"mixing"`` (Hadamard), ``"adding"`` (sum),
        ``"global_only"``, ``"local_only"``.
    is_linear_branch : bool
        If True, always use local-only regardless of lgm_type.
    global_kernel : str
        ``"KSAttention"`` | ``"linear_attention"`` | ``"classical_attention"``.
    embedding_type : str
        ``"spectral"`` or ``"physical"``.
    use_scale_decomposition : bool
        Whether to truncate Fourier modes.
    kernel_multiplier : int
        Q/K dimension multiplier. Default: 3.
    scaling_factor : float
        Default: 1.0.
    use_rope : bool
        Default: True.
    """

    def __init__(
        self,
        dim: int,
        modes: Tuple[int, int],
        heads: int,
        dim_head: int,
        local_kernel: str,
        kernel_size: int,
        lgm_type: str = "mixing",
        is_linear_branch: bool = False,
        global_kernel: str = "KSAttention",
        embedding_type: str = "spectral",
        use_scale_decomposition: bool = True,
        kernel_multiplier: int = 3,
        scaling_factor: float = 1.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.lgm_type = lgm_type
        self.is_linear_branch = is_linear_branch
        self.global_kernel = global_kernel
        self.local_kernel_type = local_kernel

        # Determine what we actually need
        needs_local = is_linear_branch or lgm_type in ("mixing", "adding", "local_only")
        needs_global = (not is_linear_branch) and lgm_type in ("mixing", "adding", "global_only")

        # ---- Local operator ------------------------------------------------
        if needs_local:
            if local_kernel == "mlp":
                self.local_op = MLP([dim, dim], act_fn=nn.GELU())
            elif local_kernel == "conv":
                self.local_conv = nn.Conv2d(
                    dim, dim, kernel_size, stride=1,
                    padding=kernel_size // 2, padding_mode="zeros")
            else:
                raise ValueError(f"Unknown local_kernel: {local_kernel}")

        # ---- Global operator -----------------------------------------------
        if needs_global:
            if global_kernel == "KSAttention":
                self.dynamics_embedding = DynamicsEmbedding(
                    dim=dim, modes=modes,
                    embedding_type=embedding_type,
                    use_scale_decomposition=use_scale_decomposition)
                self.kronecker_attention = KroneckerAttention(
                    dim=dim, dim_head=dim_head, heads=heads, dim_out=dim,
                    kernel_multiplier=kernel_multiplier,
                    scaling_factor=scaling_factor, use_rope=use_rope)
            elif global_kernel == "linear_attention":
                self.alt_attention = LinearAttentionBlock(
                    dim=dim, heads=heads, dim_head=dim_head)
            elif global_kernel == "classical_attention":
                self.alt_attention = ClassicalAttentionBlock(
                    dim=dim, heads=heads, dim_head=dim_head)
            else:
                raise ValueError(f"Unknown global_kernel: {global_kernel}")

        self._needs_local = needs_local
        self._needs_global = needs_global

    def _apply_local(self, c: Tensor) -> Tensor:
        if self.local_kernel_type == "mlp":
            return self.local_op(c)
        else:
            return self.local_conv(c.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def _apply_global(self, c: Tensor) -> Tensor:
        if self.global_kernel == "KSAttention":
            u_embedded = self.dynamics_embedding(c)
            return self.kronecker_attention(u_embedded)
        else:
            return self.alt_attention(c)

    def forward(self, c: Tensor) -> Tensor:
        """``[B, H, W, C] → [B, H, W, C]``."""
        if self.is_linear_branch:
            return self._apply_local(c)

        if self.lgm_type == "mixing":
            return self._apply_local(c) * self._apply_global(c)
        elif self.lgm_type == "adding":
            return self._apply_local(c) + self._apply_global(c)
        elif self.lgm_type == "global_only":
            return self._apply_global(c)
        elif self.lgm_type == "local_only":
            return self._apply_local(c)
        else:
            raise ValueError(f"Unknown lgm_type: {self.lgm_type}")


class FullScaleDynamicsLayer(nn.Module):
    r"""Full-Scale Dynamics Layer (FSDL).

    .. math::
        \mathscr{F}_\theta(v) = \Psi_\theta\bigl(\sum_j \text{LGM}^{(\text{nl})}_j\bigr)
        + \sum_i \text{LGM}^{(\text{lin})}_i

    Parameters
    ----------
    dim, num_nonlinear, num_linear, modes, heads, dim_head, local_kernel,
    kernel_size : core parameters.
    lgm_type, global_kernel, embedding_type, use_scale_decomposition :
        ablation parameters passed through to each LGM_Transformation.
    """

    def __init__(
        self,
        dim: int,
        num_nonlinear: int,
        num_linear: int,
        modes: Tuple[int, int],
        heads: int,
        dim_head: int,
        local_kernel: str,
        kernel_size: int,
        lgm_type: str = "mixing",
        global_kernel: str = "KSAttention",
        embedding_type: str = "spectral",
        use_scale_decomposition: bool = True,
        kernel_multiplier: int = 3,
        scaling_factor: float = 1.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.num_nonlinear = num_nonlinear
        self.num_linear = num_linear

        common = dict(
            dim=dim, modes=modes, heads=heads, dim_head=dim_head,
            local_kernel=local_kernel, kernel_size=kernel_size,
            lgm_type=lgm_type, global_kernel=global_kernel,
            embedding_type=embedding_type,
            use_scale_decomposition=use_scale_decomposition,
            kernel_multiplier=kernel_multiplier,
            scaling_factor=scaling_factor, use_rope=use_rope)

        self.nonlinear_branches = nn.ModuleList(
            [LGM_Transformation(is_linear_branch=False, **common)
             for _ in range(num_nonlinear)])

        self.linear_branches = nn.ModuleList(
            [LGM_Transformation(is_linear_branch=True, **common)
             for _ in range(num_linear)])

        self.psi = MLP([dim, dim], act_fn=nn.GELU())

    def forward(self, c: Tensor) -> Tensor:
        """``[B, H, W, C] → [B, H, W, C]``."""
        nl_sum = (torch.stack([b(c) for b in self.nonlinear_branches], dim=0).sum(dim=0)
                  if self.num_nonlinear > 0 else torch.zeros_like(c))
        lin_sum = (torch.stack([b(c) for b in self.linear_branches], dim=0).sum(dim=0)
                   if self.num_linear > 0 else torch.zeros_like(c))
        return self.psi(nl_sum) + lin_sum


# =============================================================================
# Top-Level Model
# =============================================================================


# Architecture type aliases (ablation_params → internal)
# _ARCH_ALIASES = {
#     "only_sequential": "only_sequential",
#     "only_parallel": "only_parallel",
#     "hybrid": "hybrid",
#     # backward compat
#     "hierarchical": "only_sequential",
#     "parallel": "only_parallel",
# }


class DynFormer(nn.Module):
    r"""DynFormer — Dynamics-Informed Transformer Neural Operator.

    Supports all ablation properties from ``ablation_params.py``:

    - ``global_kernel``: ``'KSAttention'`` | ``'linear_attention'`` | ``'classical_attention'``
    - ``embedding_type``: ``'spectral'`` | ``'physical'``
    - ``use_scale_decomposition``: bool
    - ``lgm_type``: ``'mixing'`` | ``'adding'`` | ``'global_only'`` | ``'local_only'``
    - ``architecture_type``: ``'hybrid'`` | ``'only_sequential'`` | ``'only_parallel'``

    Parameters
    ----------
    model_config : object
        Configuration namespace.
    device : torch.device
        Target device.
    """

    def __init__(self, model_config, device: torch.device):
        super().__init__()
        self.config = model_config
        self.device = device

        # ---- Dims -----------------------------------------------------------
        self.in_dim = (
            model_config.input_dim * (model_config.inp_involve_history_step + 1)
            + model_config.coor_input_dim)
        self.out_dim = model_config.output_dim
        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test
        self.hidden_dim = model_config.hidden_dim
        self.num_layers = model_config.num_layers
        self.out_steps = getattr(model_config, "out_steps", 1)

        spectral_modes: Tuple[int, int] = tuple(model_config.spectral_modes)
        heads: int = model_config.n_head
        dim_head: int = self.hidden_dim // heads

        # ---- Ablation config (with defaults) --------------------------------
        self.architecture_type = getattr(model_config, "architecture_type", "hybrid")
        global_kernel: str = getattr(model_config, "global_kernel", "KSAttention")
        embedding_type: str = getattr(model_config, "embedding_type", "spectral")
        use_scale_decomposition: bool = getattr(model_config, "use_scale_decomposition", True)
        lgm_type: str = getattr(model_config, "lgm_type", "mixing")

        latent_multiplier = 2
        self.v_dim = self.hidden_dim * latent_multiplier

        # ---- 1. Lifting -----------------------------------------------------
        self.lifting = nn.Sequential(
            MLP([self.in_dim, self.v_dim], act_fn=nn.GELU()),
            MLP([self.v_dim, self.hidden_dim], act_fn=nn.GELU()))

        # ---- 2. Latent Evolution --------------------------------------------
        self.fsdl_layers = nn.ModuleList()
        init_dt = getattr(model_config, "evo_step_c_in_hierachy", 1.0)
        self.delta_t = nn.ParameterList()

        for _ in range(self.num_layers):
            self.fsdl_layers.append(FullScaleDynamicsLayer(
                dim=self.hidden_dim,
                num_nonlinear=model_config.num_nonlinear,
                num_linear=model_config.num_linear,
                modes=spectral_modes, heads=heads, dim_head=dim_head,
                local_kernel=model_config.local_kernel,
                kernel_size=model_config.kernel_size,
                lgm_type=lgm_type, global_kernel=global_kernel,
                embedding_type=embedding_type,
                use_scale_decomposition=use_scale_decomposition))
            self.delta_t.append(nn.Parameter(torch.tensor(float(init_dt))))

        # Recovery MLP for hybrid/parallel architectures
        if self.architecture_type in ("hybrid", "only_parallel"):
            self.recovery_mlp = MLP([self.num_layers, 1], act_fn=nn.GELU())

        self.act = nn.GELU()

        # ---- 3. Projection --------------------------------------------------
        self.projection = nn.Sequential(
            MLP([self.hidden_dim, self.v_dim], act_fn=nn.GELU()),
            MLP([self.v_dim, self.out_steps * self.out_dim], act_fn=nn.GELU()))

        self.to(self.device)

    # ---- Evolution strategies -----------------------------------------------

    def _evolve_only_sequential(self, c: Tensor) -> Tensor:
        for layer, dt in zip(self.fsdl_layers, self.delta_t):
            c = c + dt * self.act(layer(c))
        return c

    def _evolve_only_parallel(self, c: Tensor) -> Tensor:
        c_init = c
        dynamics = [self.act(layer(c_init)) for layer in self.fsdl_layers]
        f_stack = torch.stack(dynamics, dim=-1)
        return c_init + self.recovery_mlp(f_stack).squeeze(-1)

    def _evolve_hybrid(self, c: Tensor) -> Tensor:
        c_init, c_shape = c, c.shape
        dynamics_list: List[Tensor] = []
        for layer, dt in zip(self.fsdl_layers, self.delta_t):
            f_c = self.act(layer(c))
            c = c + dt * f_c
            dynamics_list.append(f_c)
        f_stack = torch.stack(dynamics_list, dim=-1)
        return c_init + self.recovery_mlp(f_stack).view(c_shape)

    # ---- Forward ------------------------------------------------------------

    def forward(self, x: Tensor, static_data: List[Tensor]) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            ``[B, C_in, H, W]``.
        static_data : List[Tensor]
            ``static_data[0]``: coordinate grid ``[B, C_coord, H, W]``.

        Returns
        -------
        Tensor
            ``[B, seq_len * out_steps, out_dim, H, W]``.
        """
        B, C, H, W = x.shape
        fx_in = x.permute(0, 2, 3, 1)
        fx_pos = static_data[0].permute(0, 2, 3, 1)

        evolve_fn = {
            "only_sequential": self._evolve_only_sequential,
            "only_parallel": self._evolve_only_parallel,
            "hybrid": self._evolve_hybrid,
        }[self.architecture_type]

        outputs: List[Tensor] = []
        for _ in range(self.seq_len):
            u = torch.cat((fx_pos, fx_in), dim=-1)
            c = self.lifting(u)
            c = evolve_fn(c)
            next_u = self.projection(c)
            outputs.append(next_u)
            fx_in = torch.cat(
                (fx_in[..., self.out_steps * self.out_dim:], next_u), dim=-1)

        return (torch.stack(outputs, dim=1)
                .permute(0, 1, 4, 2, 3)
                .reshape(B, -1, self.out_dim, H, W))


# Backward-compatibility aliases
# TDyMixOp = DynFormer
# TDyMixOp2d = DynFormer
