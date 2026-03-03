import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
import numpy as np
from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class GeAct(nn.Module):
    """Gated activation function"""
    def __init__(self, act_fn):
        super().__init__()
        self.fn = act_fn

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c//2)]) * x[..., int(c//2):]


class MLP(nn.Module):
    def __init__(self, dims, act_fn, dropout=0.):
        super().__init__()
        layers = []

        for i in range(len(dims) - 1):
            if isinstance(act_fn, GeAct) and i < len(dims) - 2:
                layers.append(nn.Linear(dims[i], dims[i+1] * 2))
            else:
                layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(act_fn)
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PoolingReducer(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim):
        super().__init__()
        self.to_in = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_ffn = PreNorm(in_dim, MLP([hidden_dim, hidden_dim, out_dim], GeAct(nn.GELU())))

    def forward(self, x):
        # note that the dimension to be pooled will be the last dimension
        # x: b nx ... c
        x = self.to_in(x)
        # pool all spatial dimension but the first one
        ndim = len(x.shape)
        x = x.mean(dim=tuple(range(2, ndim-1)))
        x = self.out_ffn(x)
        return x  # b nx c


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

class LowRankKernel(nn.Module):
    # low rank kernel, ideally operates only on one dimension
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 positional_embedding='rotary',
                 pos_dim=1,
                 normalize=False,
                 softmax=False,
                 residual=True,
                 dropout=0,
                 scaling=1,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.dim_head = dim_head
        self.heads = heads
        self.normalize = normalize
        self.residual = residual
        if dropout > 1e-6:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        self.to_q = nn.Linear(dim, dim_head*heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head*heads, bias=False)

        assert positional_embedding in ['rff', 'rotary', 'learnable', 'none']
        self.positional_embedding = positional_embedding
        self.pos_dim = pos_dim

        if positional_embedding == 'rff':
            self.pos_emb = GaussianFourierFeatureTransform(pos_dim, dim_head, scale=1,
                                                           learnable=False, num_heads=heads)
        elif positional_embedding == 'rotary':
            self.pos_emb = RotaryEmbedding(dim_head//self.pos_dim, min_freq=1/64)
        elif positional_embedding == 'learnable':
            self.pos_emb = nn.Sequential(
                GaussianFourierFeatureTransform(pos_dim, dim_head * heads // 2, scale=1,
                                                learnable=True),
                nn.Linear(dim_head * heads // 2, dim_head*heads, bias=False),
                nn.GELU(),
                nn.Linear(dim_head*heads, dim_head*heads, bias=False))
        else:
            pass
        self.init_gain = 0.02   # 1 / np.sqrt(dim_head)
        # self.diagonal_weight = nn.Parameter(1 / np.sqrt(dim_head) *
        #                                     torch.ones(heads, 1, 1), requires_grad=True)
        self.initialize_qk_weights()
        self.softmax = softmax

        self.residual = residual
        if self.residual:
            self.gamma = nn.Parameter(torch.tensor(1 / np.sqrt(dim_head)), requires_grad=True)
        else:
            self.gamma = 0
        self.scaling = scaling

    def initialize_qk_weights(self):
        xavier_uniform_(self.to_q.weight, gain=self.init_gain)
        xavier_uniform_(self.to_k.weight, gain=self.init_gain)
        # torch.nn.init.normal_(self.to_q.weight, std=self.init_gain)
        # torch.nn.init.normal_(self.to_k.weight, std=self.init_gain)

    def normalize_wrt_domain(self, x):
        x = (x - x.mean(dim=-2, keepdim=True)) / (x.std(dim=-2, keepdim=True) + 1e-5)
        return x

    def forward(self, u_x, u_y=None, pos_x=None, pos_y=None):
        # u_x, u_y: b n c
        # u_x is from the first source
        # u_y is from the second source
        # pos: b n d
        if u_y is None:
            u_y = u_x

        n = u_y.shape[1]

        q = self.to_q(u_x)
        k = self.to_k(u_y)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        if self.normalize:
            q = self.normalize_wrt_domain(q)
            k = self.normalize_wrt_domain(k)

        if self.positional_embedding != 'none' and pos_x is None:
            raise ValueError('positional embedding is not none but pos is None')

        if self.positional_embedding != 'rotary' and \
                self.positional_embedding != 'none' and \
                self.positional_embedding != 'rff':
            pos_x_emb = self.pos_emb(pos_x)
            if pos_y is None:
                pos_y_emb = pos_x_emb
            else:
                pos_y_emb = self.pos_emb(pos_y)
            q = q * pos_x_emb
            k = k * pos_y_emb
        elif self.positional_embedding == 'rff':

            pos_x_emb = self.pos_emb(pos_x, unfold_head=True)
            if pos_y is None:
                pos_y_emb = pos_x_emb
            else:
                pos_y_emb = self.pos_emb(pos_y, unfold_head=True)

            # duplicate q, k
            q_ = torch.cat((q, q), dim=-1)
            k_ = torch.cat((k, k), dim=-1)
            q = q_ * pos_x_emb
            k = k_ * pos_y_emb

        elif self.positional_embedding == 'rotary':
            if self.pos_dim == 2:
                assert pos_x.shape[-1] == 2
                q_freqs_x = self.pos_emb.forward(pos_x[..., 0], q.device)
                q_freqs_y = self.pos_emb.forward(pos_x[..., 1], q.device)
                q_freqs_x = repeat(q_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                q_freqs_y = repeat(q_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                if pos_y is None:
                    k_freqs_x = q_freqs_x
                    k_freqs_y = q_freqs_y
                else:
                    k_freqs_x = self.pos_emb.forward(pos_y[..., 0], k.device)
                    k_freqs_y = self.pos_emb.forward(pos_y[..., 1], k.device)
                    k_freqs_x = repeat(k_freqs_x, 'b n d -> b h n d', h=k.shape[1])
                    k_freqs_y = repeat(k_freqs_y, 'b n d -> b h n d', h=k.shape[1])

                q = apply_2d_rotary_pos_emb(q, q_freqs_x, q_freqs_y)
                k = apply_2d_rotary_pos_emb(k, k_freqs_x, k_freqs_y)
            elif self.pos_dim == 1:
                assert pos_x.shape[-1] == 1

                q_freqs = self.pos_emb.forward(pos_x[..., 0], q.device).unsqueeze(0)
                q_freqs = repeat(q_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                if pos_y is None:
                    k_freqs = q_freqs
                else:
                    k_freqs = self.pos_emb.forward(pos_y[..., 0], k.device).unsqueeze(0)
                    k_freqs = repeat(k_freqs, '1 n d -> b h n d', b=q.shape[0], h=q.shape[1])

                q = apply_rotary_pos_emb(q, q_freqs)
                k = apply_rotary_pos_emb(k, k_freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        else:  # do nothing
            pass

        K = torch.einsum('bhid,bhjd->bhij', q, k) * self.scaling  # if not on uniform grid, need to consider quadrature weights
        K = self.dropout(K)
        if self.softmax:
            K = F.softmax(K, dim=-1)
        if self.residual:
            K = K + self.gamma * torch.eye(n).to(q.device).view(1, 1, n, n) / n
        return K


class GroupNorm(nn.Module):
    # group norm with channel at the last dimension
    def __init__(self, num_groups, num_channels,
                 domain_wise=False,
                 eps=1e-8, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.domain_wise = domain_wise
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels), requires_grad=True)
            self.bias = nn.Parameter(torch.zeros(num_channels), requires_grad=True)

    def forward(self, x):
        # b h w c
        b, g_c = x.shape[0], x.shape[-1]
        c = g_c // self.num_groups
        if self.domain_wise:
            x = rearrange(x, 'b ... (g c) -> b g (... c)', g=self.num_groups)
        else:
            x = rearrange(x, 'b ... (g c) -> b ... g c', g=self.num_groups)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x = (x - mean) / (var + self.eps).sqrt()
        if self.domain_wise:
            # b g (... c) -> b ... (g c)
            x = x.view(b, self.num_groups, -1, c)
            x = rearrange(x, 'b g ... c -> b ... (g c)')
        else:
            x = rearrange(x, 'b ... g c -> b ... (g c)',
                          g=self.num_groups)
        if self.affine:
            x = x * self.weight + self.bias
        return x


class FABlock2D(nn.Module):
    # contains factorization and attention on each axis
    def __init__(self,
                 dim,
                 dim_head,
                 latent_dim,
                 heads,
                 dim_out,
                 use_rope=True,
                 kernel_multiplier=3,
                 scaling_factor=1.0):
        super().__init__()

        self.dim = dim
        self.latent_dim = latent_dim
        self.heads = heads
        self.dim_head = dim_head
        self.in_norm = nn.LayerNorm(dim)
        self.to_v = nn.Linear(self.dim, heads * dim_head, bias=False)
        self.to_in = nn.Linear(self.dim, self.dim, bias=False)

        self.to_x = nn.Sequential(
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )
        self.to_y = nn.Sequential(
            Rearrange('b nx ny c -> b ny nx c'),
            PoolingReducer(self.dim, self.dim, self.latent_dim),
        )

        positional_encoding = 'rotary' if use_rope else 'none'
        use_softmax = False
        self.low_rank_kernel_x = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,  # add a diagonal bias
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)
        self.low_rank_kernel_y = LowRankKernel(self.latent_dim, dim_head * kernel_multiplier, heads,
                                               positional_embedding=positional_encoding,
                                               residual=False,
                                               softmax=use_softmax,
                                               scaling=1 / np.sqrt(dim_head * kernel_multiplier)
                                               if kernel_multiplier > 4 or use_softmax else scaling_factor)

        self.to_out = nn.Sequential(
            GroupNorm(heads, dim_head * heads, domain_wise=False, affine=False),
            nn.Linear(dim_head * heads, dim_out, bias=False),
            nn.GELU(),
            nn.Linear(dim_out, dim_out, bias=False))

    def forward(self, u, pos_lst):
        # x: b c h w
        u = self.in_norm(u)
        v = self.to_v(u)
        u = self.to_in(u)

        u_x = self.to_x(u)
        u_y = self.to_y(u)

        pos_x, pos_y = pos_lst
        k_x = self.low_rank_kernel_x(u_x, pos_x=pos_x)
        k_y = self.low_rank_kernel_y(u_y, pos_x=pos_y)

        u_phi = rearrange(v, 'b i l (h c) -> b h i l c', h=self.heads)
        u_phi = torch.einsum('bhij,bhjmc->bhimc', k_x, u_phi)
        u_phi = torch.einsum('bhlm,bhimc->bhilc', k_y, u_phi)
        u_phi = rearrange(u_phi, 'b h i l c -> b i l (h c)', h=self.heads)
        return self.to_out(u_phi)


# Gaussian Fourier features
# code modified from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels,
                 mapping_size=256, scale=10, learnable=False,
                 num_heads=1):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size

        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size * num_heads)) * scale,
                               requires_grad=learnable)
        self.num_heads = num_heads

    def forward(self, x, unfold_head=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        if unfold_head:
            x = rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads)
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class FactorizedTransformer(nn.Module):
    def __init__(self,
                 dim,
                 dim_head,
                 heads,
                 dim_out,
                 depth,
                 **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):

            layer = nn.ModuleList([])
            layer.append(nn.Sequential(
                GaussianFourierFeatureTransform(2, dim // 2, 8),
                nn.Linear(dim, dim)
            ))
            layer.append(FABlock2D(dim, dim_head, dim, heads, dim_out, use_rope=True, **kwargs))
            self.layers.append(layer)

    def forward(self, u, pos_lst):
        b, nx, ny, c = u.shape
        nx, ny = pos_lst[0].shape[0], pos_lst[1].shape[0]
        pos = torch.stack(torch.meshgrid([pos_lst[0].squeeze(-1), pos_lst[1].squeeze(-1)]), dim=-1)
        for pos_enc, attn_layer in self.layers:
            u = u + pos_enc(pos).view(1, nx, ny, -1)
            u = attn_layer(u, pos_lst) + u
        return u


class FactFormer(nn.Module):
    def __init__(self,
                 model_config, device
                 ):
        super().__init__()
        self.config = model_config
        self.device = device
        # self.resolutions = config.resolutions   # hierachical resolutions, [16, 8, 4]
        # self.out_resolution = config.out_resolution

        self.in_dim = model_config.input_dim * (model_config.inp_involve_history_step + 1) + model_config.coor_input_dim
        self.out_dim = model_config.output_dim
        self.seq_len = model_config.ar_nseq_train + model_config.ar_nseq_test
        self.dim = model_config.hidden_dim                 # dimension of the transformer
        self.depth = model_config.num_layers // 2           # depth of the encoder transformer
        self.propogator_depth = model_config.num_layers // 2
        self.dim_head = model_config.hidden_dim // model_config.n_head

        self.heads = model_config.n_head

        self.kernel_multiplier = model_config.kernel_multiplier
        self.latent_multiplier = model_config.latent_multiplier
        self.latent_dim = int(self.dim * self.latent_multiplier)
        self.out_steps = model_config.out_steps

        # flatten time window
        self.to_in = nn.Linear(self.in_dim, self.dim, bias=True).to(device)

        # assume input is b c t h w d
        self.encoder = FactorizedTransformer(self.dim, self.dim_head, self.heads, self.dim, self.depth,
                                             kernel_multiplier=self.kernel_multiplier).to(device)
        self.expand_latent = nn.Linear(self.dim, self.latent_dim, bias=False).to(device)
        self.out_steps_latent_time_emb = nn.Parameter(torch.randn(1, self.out_steps,
                                                        1, 1, self.latent_dim) * 0.02,
                                            requires_grad=True).to(device)

        self.propagator = PreNorm(self.latent_dim,
                                  MLP([self.latent_dim for _ in range(self.propogator_depth)], act_fn=nn.GELU())).to(device)
        self.simple_to_out = nn.Sequential(
            Rearrange('b nx ny c -> b c (nx ny)'),
            nn.GroupNorm(num_groups=int(8 * self.latent_multiplier), num_channels=self.latent_dim),
            nn.Conv1d(self.latent_dim, self.dim, kernel_size=1, stride=1, padding=0,
                      groups=8, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim, self.dim // 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.Conv1d(self.dim // 2, self.out_dim, kernel_size=1, stride=1, padding=0, bias=True)
        ).to(device)

    # def forward(self,
    #             u,
    #             pos_lst,
    #             latent_steps=1,
    #             ):
    #     b, nx, ny, c = u.shape
    #     u = self.to_in(u)
    #     u = self.encoder(u, pos_lst)
    #     # u = self.temporal_conv(u)
    #     u_lst = []
    #     u = self.expand_latent(u)
    #     for l_t in range(latent_steps):
    #         u = u + self.latent_time_emb[:, l_t, ...]
    #         u = self.propagator(u) + u
    #         u_lst.append(u)

    #     u = torch.cat(u_lst, dim=0)
    #     u = self.simple_to_out(u)
    #     u = rearrange(u, '(t b) c (nx ny) -> b t nx ny c', nx=nx, ny=ny, t=latent_steps)
    #     return u


    def forward(self, x: torch.Tensor, static_data: list) -> torch.Tensor:
        """
        Forward pass with autoregressive prediction.
        
        Args:
            x: (B, T*C, H, W) - Temporal snapshots concatenated along channel dimension
            static_data: List[Tensor] - Static features (e.g., coordinates)
        
        Returns:
            (B, seq_len, C, H, W) - Predicted sequence
        """
        B, C, H, W = x.shape
        current_input = x.permute(0, 2, 3, 1)
        coor_input = static_data[0].permute(0, 2, 3, 1)

        pos_x = torch.linspace(0, 2*np.pi, H).to(self.device).unsqueeze(-1)
        pos_y = torch.linspace(0, 2*np.pi, W).to(self.device).unsqueeze(-1)
        pos_lst = [pos_x, pos_y]

        predictions = []
        for t in range(self.seq_len):
            fx = torch.cat((coor_input, current_input), -1)
            fx = self.to_in(fx)
            fx = self.encoder(fx, pos_lst)
            fx = self.expand_latent(fx)

            dec_by_latent_propagation = []
            for step in range(self.out_steps):
                fx = fx + self.out_steps_latent_time_emb[:, step, ...]
                fx = self.propagator(fx) + fx
                dec_by_latent_propagation.append(fx)
            fx = torch.cat(dec_by_latent_propagation, dim=0)  # concatenate in temporal dimension

            fx = self.simple_to_out(fx)
            fx = rearrange(fx, '(t b) c (nx ny) -> b nx ny (t c)', nx=H, ny=W, t=self.out_steps)

            predictions.append(fx)
            current_input = torch.cat((current_input[..., self.out_steps*self.out_dim:], fx), dim=-1)

        # Reshape output to (B, total_steps, C, H, W)
        predictions = torch.stack(predictions, dim=1).permute(0, 1, 4, 2, 3)
        predictions = predictions.reshape(B, -1, self.out_dim, H, W)
        
        return predictions

