"""
DynFormer: Dynamics-Informed Transformer Neural Operator for PDEs.

This module implements the DynFormer architecture, a dynamics-informed neural
operator that combines spectral processing with Kronecker-factored attention
for solving partial differential equations (PDEs).

Architectural Hierarchy
-----------------------
1. SpectralDynamicsEmbedding  — FFT-based spectral feature extraction
2. KroneckerAttention         — Factored global attention via Kronecker structure
3. LGM_Transformation         — Local-Global-Mixing (Hadamard product)
4. FullScaleDynamicsLayer      — Single FSDL block with linear + nonlinear branches
5. DynFormer                   — Top-level model: Lifting → L×FSDL → Projection
"""

import math
from typing import List, Literal, Optional, Tuple

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
        Dimensions for each layer. Length determines network depth.
    act_fn : nn.Module
        Activation function applied between hidden layers.
    dropout : float, optional
        Dropout probability between hidden layers. Default: 0.0.
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
        """
        Parameters
        ----------
        x : Tensor
            Input tensor of shape ``[..., dims[0]]``.

        Returns
        -------
        Tensor
            Output tensor of shape ``[..., dims[-1]]``.
        """
        return self.net(x)


# =============================================================================
# Positional Embedding Utilities
# =============================================================================


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) for encoding coordinate positions.

    Parameters
    ----------
    dim : int
        Embedding dimension (must be even).
    min_freq : float, optional
        Minimum frequency for the embedding. Default: 1/64.
    scale : float, optional
        Scaling factor for coordinates. Default: 1.0.
    """

    def __init__(self, dim: int, min_freq: float = 1 / 64, scale: float = 1.0):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coordinates: Tensor, device: torch.device) -> Tensor:
        """
        Parameters
        ----------
        coordinates : Tensor
            Coordinate positions of shape ``[B, N]`` or ``[N]``.
        device : torch.device
            Target device.

        Returns
        -------
        Tensor
            Frequency embeddings of shape ``[..., D]``.
        """
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum("... i , j -> ... i j", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the hidden dims of *x* for RoPE."""
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t: Tensor, freqs: Tensor) -> Tensor:
    """Apply 1-D rotary positional embedding to tensor *t*."""
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(
    t: Tensor, freqs_x: Tensor, freqs_y: Tensor
) -> Tensor:
    """Apply 2-D rotary positional embedding by splitting dimensions."""
    d = t.shape[-1]
    t_x, t_y = t[..., : d // 2], t[..., d // 2 :]
    t_x = apply_rotary_pos_emb(t_x, freqs_x)
    t_y = apply_rotary_pos_emb(t_y, freqs_y)
    return torch.cat((t_x, t_y), dim=-1)


# =============================================================================
# Core Attention Components
# =============================================================================


class _PositionAwareReductionKernel(nn.Module):
    r"""Position-aware low-rank kernel for computing attention matrices along
    a single coordinate axis.

    Produces a kernel matrix :math:`\mathcal{K} \in \mathbb{R}^{N \times N}`
    using learned Q/K projections with optional rotary positional encoding.
    This corresponds to the reduction vectors :math:`R_{Q_q}, R_{K_q}` in the
    paper that contract tensors along coordinate axes.

    Parameters
    ----------
    dim : int
        Input feature dimension.
    dim_head : int
        Dimension per attention head for Q/K projections.
    heads : int
        Number of attention heads.
    use_rope : bool, optional
        Whether to use Rotary Positional Embedding. Default: True.
    scaling : float, optional
        Scaling factor for the kernel matrix. Default: 1.0.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        use_rope: bool = True,
        scaling: float = 1.0,
    ):
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
        """Compute the kernel matrix for one coordinate axis.

        Parameters
        ----------
        u : Tensor
            Reduced features along one axis, shape ``[B, N, C]``.
        pos : Tensor, optional
            Positional coordinates, shape ``[N, 1]``.

        Returns
        -------
        Tensor
            Kernel matrix of shape ``[B, H, N, N]`` where ``H`` = heads.
        """
        q = rearrange(self.ffn_q(u), "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(self.ffn_k(u), "b n (h d) -> b h n d", h=self.heads)

        if self.pos_emb is not None and pos is not None:
            freqs = self.pos_emb(pos[..., 0], q.device).unsqueeze(0)
            freqs = repeat(freqs, "1 n d -> b h n d", b=q.shape[0], h=self.heads)
            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

        return torch.einsum("bhid,bhjd->bhij", q, k) * self.scaling


class SpectralDynamicsEmbedding(nn.Module):
    r"""Spectral Dynamics Embedding via 2-D FFT-based feature extraction.

    Performs 2-D FFT on the input, retains the lowest :math:`M_1 \times M_2`
    Fourier modes, applies learnable complex-valued spectral kernels
    :math:`W_{Q_q}, W_{K_q}, W_V` via channel-wise multiplication, and returns
    to the physical domain via inverse FFT with zero-padding.

    .. note::
        For efficiency the implementation uses a **shared** spectral kernel
        followed by separate linear projections.  This is mathematically
        equivalent to applying three independent spectral kernels.

    Parameters
    ----------
    dim : int
        Input / output channel dimension.
    modes : Tuple[int, int]
        Number of Fourier modes to retain along each spatial axis ``(M1, M2)``.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per attention head for the value projection.
    """

    def __init__(
        self,
        dim: int,
        modes: Tuple[int, int],
        heads: int,
        dim_head: int,
    ):
        super().__init__()
        self.dim = dim
        self.modes1, self.modes2 = modes
        self.heads = heads
        self.dim_head = dim_head

        # Learnable complex-valued spectral kernels
        scale = 1.0 / (dim * dim)
        self.spectral_weights_pos = nn.Parameter(
            scale
            * torch.randn(dim, dim, self.modes1, self.modes2, dtype=torch.cfloat)
        )
        self.spectral_weights_neg = nn.Parameter(
            scale
            * torch.randn(dim, dim, self.modes1, self.modes2, dtype=torch.cfloat)
        )

        # Post-spectral normalisation & feature projection
        self.norm = nn.LayerNorm(dim)
        self.to_features = nn.Linear(dim, dim, bias=False)

        # Per-axis reduction projections (for Q/K generation)
        self.to_axis_x = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim, bias=False),
        )
        self.to_axis_y = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
            nn.Linear(dim, dim, bias=False),
        )

        # Value projection
        self.to_v = nn.Linear(dim, heads * dim_head, bias=False)

    # --------------------------------------------------------------------- #

    @staticmethod
    def _compl_mul2d(inp: Tensor, weights: Tensor) -> Tensor:
        """``(B, X, Y, C_in) × (C_in, C_out, M1, M2) → (B, X, Y, C_out)``."""
        return torch.einsum("bxyi,ioxy->bxyo", inp, weights)

    # --------------------------------------------------------------------- #

    def forward(self, u: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Extract spectral dynamics features.

        Parameters
        ----------
        u : Tensor
            Input features of shape ``[B, H, W, C]``.

        Returns
        -------
        u_x : Tensor
            Axis-x reduction features, ``[B, H_spatial, C]``.
        u_y : Tensor
            Axis-y reduction features, ``[B, W_spatial, C]``.
        v : Tensor
            Value features, ``[B, H_spatial, W_spatial, heads * dim_head]``.
        """
        B, H, W, C = u.shape

        # ---- Spectral dynamics embedding ---------------------------------------- #
        u_ft = torch.fft.rfft2(u, dim=(-3, -2))

        out_ft = torch.zeros(B, H, W, C, dtype=torch.cfloat, device=u.device)
        # Positive frequency modes
        out_ft[:, : self.modes1, : self.modes2, :] = self._compl_mul2d(
            u_ft[:, : self.modes1, : self.modes2, :], self.spectral_weights_pos
        )
        # Negative frequency modes
        out_ft[:, -self.modes1 :, : self.modes2, :] = self._compl_mul2d(
            u_ft[:, -self.modes1 :, : self.modes2, :], self.spectral_weights_neg
        )
        # Passthrough band (unweighted low-frequency modes)
        out_ft[:, -self.modes1 : self.modes1, : self.modes2, :] = u_ft[
            :, -self.modes1 : self.modes1, : self.modes2, :
        ].clone()

        u = torch.fft.irfft2(out_ft, s=(H, W), dim=(-3, -2))

        # ---- Feature projections ---------------------------------------- #
        u = self.norm(u)
        u = self.to_features(u)

        v = self.to_v(u)                          # [B, H, W, heads*dim_head]
        u_x = self.to_axis_x(u.mean(dim=2))       # [B, H, C]
        u_y = self.to_axis_y(u.mean(dim=1))       # [B, W, C]
        return u_x, u_y, v


class KroneckerAttention(nn.Module):
    r"""Kronecker-Factored Attention for 2-D spatial domains.

    Employs position-aware reduction vectors :math:`R_{Q_q}, R_{K_q}` to
    contract tensors along each coordinate axis, yielding kernel matrices
    :math:`\mathcal{K}_1` and :math:`\mathcal{K}_2`.  The output is computed
    via matrix chaining:

    .. math::
        \text{out} = \mathcal{K}_1 \, \tilde{V}_c \, \mathcal{K}_2^\top

    Parameters
    ----------
    dim : int
        Input feature dimension for the reduction kernels.
    dim_head : int
        Base dimension per attention head (for the value tensor).
    heads : int
        Number of attention heads.
    dim_out : int
        Output channel dimension.
    kernel_multiplier : int, optional
        Multiplier for Q/K head dim relative to V head dim. Default: 3.
    scaling_factor : float, optional
        Scaling factor for the kernel matrices. Default: 1.0.
    use_rope : bool, optional
        Whether to use Rotary Positional Embedding. Default: True.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        heads: int,
        dim_out: int,
        kernel_multiplier: int = 3,
        scaling_factor: float = 1.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        qk_dim_head = dim_head * kernel_multiplier

        # Reduction kernels for each spatial axis
        self.kernel_x = _PositionAwareReductionKernel(
            dim, qk_dim_head, heads, use_rope=use_rope, scaling=scaling_factor
        )
        self.kernel_y = _PositionAwareReductionKernel(
            dim, qk_dim_head, heads, use_rope=use_rope, scaling=scaling_factor
        )

        # Output projection
        self.to_out = nn.Sequential(
            nn.LayerNorm(dim_head * heads),
            nn.Linear(dim_head * heads, dim_out, bias=False),
        )

    def forward(self, u_x: Tensor, u_y: Tensor, v: Tensor) -> Tensor:
        r"""Compute Kronecker-factored attention.

        Parameters
        ----------
        u_x : Tensor
            Axis-x reduction features, shape ``[B, H_s, C]``.
        u_y : Tensor
            Axis-y reduction features, shape ``[B, W_s, C]``.
        v : Tensor
            Value features, shape ``[B, H_s, W_s, heads * dim_head]``.

        Returns
        -------
        Tensor
            Attention output of shape ``[B, H_s, W_s, dim_out]``.
        """
        H_s, W_s = u_x.shape[1], u_y.shape[1]

        # Positional coordinates
        pos_x = torch.linspace(0, 2 * math.pi, H_s, device=u_x.device).unsqueeze(-1)
        pos_y = torch.linspace(0, 2 * math.pi, W_s, device=u_y.device).unsqueeze(-1)

        # Kernel matrices  K_1: [B, heads, H, H],  K_2: [B, heads, W, W]
        k_1 = self.kernel_x(u_x, pos=pos_x)
        k_2 = self.kernel_y(u_y, pos=pos_y)

        # Multi-head value: [B, heads, H, W, dim_head]
        v_mh = rearrange(v, "b i l (h c) -> b h i l c", h=self.heads)

        # Matrix chaining: K_1 @ V @ K_2^T
        out = torch.einsum("bhij,bhjmc->bhimc", k_1, v_mh)
        out = torch.einsum("bhlm,bhimc->bhilc", k_2, out)

        out = rearrange(out, "b h i l c -> b i l (h c)")
        return self.to_out(out)


# =============================================================================
# Layer Modules
# =============================================================================


class LGM_Transformation(nn.Module):
    r"""Local-Global-Mixing (LGM) Transformation.

    Computes the Hadamard (element-wise) product of a **global** operator
    and a **local** operator when ``mode="nonlinear"``.  When
    ``mode="linear"``, only the local (pointwise) operator is applied.

    * **Global operator** — :class:`SpectralDynamicsEmbedding` +
      :class:`KroneckerAttention`.
    * **Local operator** — pointwise MLP or 1×1 convolution (mesh-invariant).

    Parameters
    ----------
    dim : int
        Input / output channel dimension.
    modes : Tuple[int, int]
        Fourier modes for the spectral embedding ``(M1, M2)``.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per attention head.
    local_kernel : str
        Type of local operator: ``"mlp"`` or ``"conv"``.
    kernel_size : int
        Conv kernel size (only used when ``local_kernel="conv"``).
    mode : str
        ``"nonlinear"`` (Hadamard of local × global) or ``"linear"``
        (local only).
    kernel_multiplier : int, optional
        Q/K dimension multiplier for :class:`KroneckerAttention`. Default: 3.
    scaling_factor : float, optional
        Scaling for the attention kernel. Default: 1.0.
    use_rope : bool, optional
        Whether to use RoPE in the attention kernel. Default: True.
    """

    def __init__(
        self,
        dim: int,
        modes: Tuple[int, int],
        heads: int,
        dim_head: int,
        local_kernel: str,
        kernel_size: int,
        mode: str = "nonlinear",
        kernel_multiplier: int = 3,
        scaling_factor: float = 1.0,
        use_rope: bool = True,
    ):
        super().__init__()
        assert mode in ("nonlinear", "linear"), f"Unknown mode: {mode}"
        self.mode = mode
        self.local_kernel_type = local_kernel

        # ---- Local operator ------------------------------------------------
        if local_kernel == "mlp":
            self.local_op = MLP([dim, dim], act_fn=nn.GELU())
        elif local_kernel == "conv":
            self.local_conv = nn.Conv2d(
                dim, dim, kernel_size, stride=1,
                padding=kernel_size // 2, padding_mode="zeros",
            )
        else:
            raise ValueError(f"Unknown local_kernel: {local_kernel}")

        # ---- Global operator (only for nonlinear mode) ---------------------
        if mode == "nonlinear":
            self.spectral_embedding = SpectralDynamicsEmbedding(
                dim=dim, modes=modes, heads=heads, dim_head=dim_head,
            )
            self.kronecker_attention = KroneckerAttention(
                dim=dim, dim_head=dim_head, heads=heads, dim_out=dim,
                kernel_multiplier=kernel_multiplier,
                scaling_factor=scaling_factor, use_rope=use_rope,
            )

    # --------------------------------------------------------------------- #

    def _apply_local(self, c: Tensor) -> Tensor:
        """Apply the local operator.  Shape: ``[B, H, W, C] → [B, H, W, C]``."""
        if self.local_kernel_type == "mlp":
            return self.local_op(c)
        else:  # conv
            return self.local_conv(c.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def _apply_global(self, c: Tensor) -> Tensor:
        """Apply the global operator.  Shape: ``[B, H, W, C] → [B, H, W, C]``."""
        u_x, u_y, v = self.spectral_embedding(c)
        return self.kronecker_attention(u_x, u_y, v)

    # --------------------------------------------------------------------- #

    def forward(self, c: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        c : Tensor
            Input features of shape ``[B, H, W, C]``.

        Returns
        -------
        Tensor
            Transformed output of shape ``[B, H, W, C]``.
        """
        if self.mode == "nonlinear":
            return self._apply_local(c) * self._apply_global(c)
        else:  # linear
            return self._apply_local(c)


class FullScaleDynamicsLayer(nn.Module):
    r"""Full-Scale Dynamics Layer (FSDL).

    A single FSDL block that computes:

    .. math::
        \mathscr{F}_\theta(v) = \Psi_\theta\!\bigl(\textstyle\sum_{j=1}^{n_n}
        \text{LGM}^{(\text{nl})}_j(v)\bigr)
        + \textstyle\sum_{i=1}^{n_l} \text{LGM}^{(\text{lin})}_i(v)

    where :math:`\Psi_\theta` is a learnable residual refinement MLP.

    Parameters
    ----------
    dim : int
        Channel dimension.
    num_nonlinear : int
        Number of nonlinear LGM branches (:math:`n_n`).
    num_linear : int
        Number of linear LGM branches (:math:`n_l`).
    modes : Tuple[int, int]
        Fourier modes ``(M1, M2)``.
    heads : int
        Number of attention heads.
    dim_head : int
        Dimension per head.
    local_kernel : str
        ``"mlp"`` or ``"conv"``.
    kernel_size : int
        Conv kernel size (only used with ``"conv"``).
    kernel_multiplier : int, optional
        Default: 3.
    scaling_factor : float, optional
        Default: 1.0.
    use_rope : bool, optional
        Default: True.
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
        kernel_multiplier: int = 3,
        scaling_factor: float = 1.0,
        use_rope: bool = True,
    ):
        super().__init__()
        self.num_nonlinear = num_nonlinear
        self.num_linear = num_linear

        common_kwargs = dict(
            dim=dim, modes=modes, heads=heads, dim_head=dim_head,
            local_kernel=local_kernel, kernel_size=kernel_size,
            kernel_multiplier=kernel_multiplier,
            scaling_factor=scaling_factor, use_rope=use_rope,
        )

        # n_n nonlinear LGM branches  (Hadamard of local × global)
        self.nonlinear_branches = nn.ModuleList(
            [LGM_Transformation(mode="nonlinear", **common_kwargs)
             for _ in range(num_nonlinear)]
        )

        # n_l linear LGM branches  (local only)
        self.linear_branches = nn.ModuleList(
            [LGM_Transformation(mode="linear", **common_kwargs)
             for _ in range(num_linear)]
        )

        # Ψ_θ : learnable residual refinement (pointwise MLP)
        self.psi = MLP([dim, 2 * dim, dim], act_fn=nn.GELU())

    def forward(self, c: Tensor) -> Tensor:
        r"""
        Parameters
        ----------
        c : Tensor
            Input latent features of shape ``[B, H, W, C]``.

        Returns
        -------
        Tensor
            Dynamics output :math:`\mathscr{F}_\theta(v)` of shape
            ``[B, H, W, C]``.
        """
        # Nonlinear branch sum
        if self.num_nonlinear > 0:
            nl_sum = torch.stack(
                [branch(c) for branch in self.nonlinear_branches], dim=0
            ).sum(dim=0)
        else:
            nl_sum = torch.zeros_like(c)

        # Linear branch sum
        if self.num_linear > 0:
            lin_sum = torch.stack(
                [branch(c) for branch in self.linear_branches], dim=0
            ).sum(dim=0)
        else:
            lin_sum = torch.zeros_like(c)

        return self.psi(nl_sum) + lin_sum


# =============================================================================
# Top-Level Model
# =============================================================================


class DynFormer(nn.Module):
    r"""DynFormer — Dynamics-Informed Transformer Neural Operator.

    Top-level model that applies:

    1. **LiftingTransformation** — pointwise MLP projecting input fields to a
       latent channel dimension.
    2. **LatentEvolution** — a stack of *L* :class:`FullScaleDynamicsLayer`
       blocks with learnable temporal scaling :math:`\Delta t_{\theta_l}`.
    3. **ProjectionTransformation** — pointwise MLP mapping from latent
       dimension back to the output dimension.

    The ``architecture_type`` argument controls how the FSDL layers are
    composed:

    * ``"hierarchical"`` (default):
      :math:`v_l = v_{l-1} + \Delta t_{\theta_l} \mathscr{F}_{\theta_l}(v_{l-1})`
    * ``"parallel"``:
      :math:`v_L = v_0 + \sum_l \Delta t_{\theta_l} \mathscr{F}_{\theta_l}(v_0)`
    * ``"hybrid"``: hierarchical intermediate updates with a learned MLP
      recovery from collected dynamics.

    Parameters
    ----------
    model_config : object
        Configuration namespace with model hyper-parameters.
    device : torch.device
        Device to place the model on.
    """

    def __init__(self, model_config, device: torch.device):
        super().__init__()
        self.config = model_config
        self.device = device

        # ---- Derived dimensions -------------------------------------------
        self.in_dim = (
            model_config.input_dim * (model_config.inp_involve_history_step + 1)
            + model_config.coor_input_dim
        )
        self.out_dim = model_config.output_dim
        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test
        self.hidden_dim = model_config.hidden_dim
        self.num_layers = model_config.num_layers
        self.out_steps = model_config.out_steps

        spectral_modes: List[int, int] = model_config.spectral_modes
        heads: int = model_config.n_head
        dim_head: int = self.hidden_dim // heads

        # Architecture type
        self.architecture_type: str = getattr(
            model_config, "architecture_type", "hierarchical"
        )

        latent_multiplier = 2
        self.v_dim = self.hidden_dim * latent_multiplier

        # ---- 1. Lifting Transformation ------------------------------------
        self.lifting = nn.Sequential(
            MLP([self.in_dim, self.v_dim], act_fn=nn.GELU()),
            MLP([self.v_dim, self.hidden_dim], act_fn=nn.GELU()),
        )

        # ---- 2. Latent Evolution ------------------------------------------
        self.fsdl_layers = nn.ModuleList()
        init_dt = getattr(model_config, "evo_step_c_in_hierachy", 1.0)
        self.delta_t = nn.ParameterList()

        for _ in range(self.num_layers):
            self.fsdl_layers.append(
                FullScaleDynamicsLayer(
                    dim=self.hidden_dim,
                    num_nonlinear=model_config.num_nonlinear,
                    num_linear=model_config.num_linear,
                    modes=spectral_modes,
                    heads=heads,
                    dim_head=dim_head,
                    local_kernel=model_config.local_kernel,
                    kernel_size=model_config.kernel_size,
                )
            )
            self.delta_t.append(
                nn.Parameter(torch.tensor(float(init_dt)))
            )

        # Hybrid-mode recovery MLP (only used when architecture_type="hybrid")
        if self.architecture_type == "hybrid":
            self.recovery_mlp = MLP(
                [self.num_layers, 1], act_fn=nn.GELU()
            )

        # ---- 3. Projection Transformation ---------------------------------
        self.projection = nn.Sequential(
            MLP([self.hidden_dim, self.v_dim], act_fn=nn.GELU()),
            MLP(
                [self.v_dim, self.out_steps * self.out_dim], act_fn=nn.GELU()
            ),
        )

        # Move the entire model and its submodules to the specified device
        self.to(self.device)

    # --------------------------------------------------------------------- #

    def _evolve_hierarchical(self, c: Tensor) -> Tensor:
        r"""Hierarchical: :math:`v_l = v_{l-1} + \Delta t_l F_l(v_{l-1})`."""
        for i, (layer, dt) in enumerate(zip(self.fsdl_layers, self.delta_t)):
            c = c + dt * layer(c)
        return c

    def _evolve_parallel(self, c: Tensor) -> Tensor:
        r"""Parallel: :math:`v_L = v_0 + \sum_l \Delta t_l F_l(v_0)`."""
        c_init = c
        accum = torch.zeros_like(c)
        for layer, dt in zip(self.fsdl_layers, self.delta_t):
            accum = accum + dt * layer(c_init)
        return c_init + accum

    def _evolve_hybrid(self, c: Tensor) -> Tensor:
        """Hybrid: hierarchical updates + learned MLP recovery."""
        c_init = c
        dynamics_list: List[Tensor] = []
        for layer, dt in zip(self.fsdl_layers, self.delta_t):
            f_c = layer(c)
            c = c + dt * f_c
            dynamics_list.append(f_c)
        # Learned recovery from collected dynamics
        f_stack = torch.stack(dynamics_list, dim=-1)  # [..., L]
        c = c_init + self.recovery_mlp(f_stack).squeeze(-1)
        return c

    # --------------------------------------------------------------------- #

    def forward(
        self, x: Tensor, static_data: List[Tensor]
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input fields of shape ``[B, C_in, H, W]`` where
            ``C_in = input_dim * (history + 1)``.
        static_data : List[Tensor]
            List containing coordinate grids.  ``static_data[0]`` has shape
            ``[B, C_coord, H, W]``.

        Returns
        -------
        Tensor
            Predicted output of shape ``[B, seq_len * out_steps, out_dim, H, W]``.
        """
        B, C, H, W = x.shape
        # Permute to channel-last: [B, H, W, C]
        fx_in = x.permute(0, 2, 3, 1)
        fx_pos = static_data[0].permute(0, 2, 3, 1)

        # Select evolution function
        evolve_fn = {
            "hierarchical": self._evolve_hierarchical,
            "parallel": self._evolve_parallel,
            "hybrid": self._evolve_hybrid,
        }[self.architecture_type]

        outputs: List[Tensor] = []
        for _ in range(self.seq_len):
            u = torch.cat((fx_pos, fx_in), dim=-1)

            # Lifting → Latent Evolution → Projection
            c = self.lifting(u)
            c = evolve_fn(c)
            next_u = self.projection(c)

            outputs.append(next_u)
            fx_in = torch.cat(
                (fx_in[..., self.out_steps * self.out_dim :], next_u), dim=-1
            )

        # Stack and reshape: [B, seq_len, H, W, out_steps*out_dim]
        #                   → [B, seq_len*out_steps, out_dim, H, W]
        result = (
            torch.stack(outputs, dim=1)
            .permute(0, 1, 4, 2, 3)
            .reshape(B, -1, self.out_dim, H, W)
        )
        return result


# Backward-compatibility alias
TDyMixOp = DynFormer
