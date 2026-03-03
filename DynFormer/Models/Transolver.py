import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange, repeat

ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'softplus': nn.Softplus, 'ELU': nn.ELU, 'silu': nn.SiLU}


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    ## for structured mesh in 2D space
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64, H=101, W=31, kernel=3):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).contiguous().permute(0, 3, 1, 2).contiguous()  # B C H W

        ### (1) Slice
        fx_mid = self.in_project_fx(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).permute(0, 2, 3, 1).contiguous().reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5))  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_block(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
            H=85,
            W=85
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Structured_Mesh_2D(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                         dropout=dropout, slice_num=slice_num, H=H, W=W)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver(nn.Module):
    def __init__(self, model_config, device):
        super(Transolver, self).__init__()

        self.space_dim=model_config.coor_input_dim
        self.n_layers=model_config.num_layers
        self.n_hidden=model_config.hidden_dim
        self.dropout=model_config.dropout # 0
        self.n_head=model_config.n_head
        self.Time_Input=model_config.Time_Input # false
        self.act=model_config.act
        self.mlp_ratio=model_config.mlp_ratio # 1
        self.fun_dim=model_config.input_dim * (model_config.inp_involve_history_step + 1)
        self.out_dim=model_config.output_dim
        self.inp_dim=model_config.input_dim 
        self.slice_num=model_config.slice_num # 32
        self.ref=model_config.ref # 8
        self.unified_pos=model_config.unified_pos # 1
        self.H=model_config.H # 64
        self.W=model_config.W # 64
        self.seq_len=model_config.ar_nseq_train + model_config.ar_nseq_test
        self.out_steps = getattr(model_config, 'out_steps', 1)

        self.__name__ = 'Transolver_2D'

        if self.unified_pos:
            self.pos = self.get_grid().to(device)
            self.preprocess = MLP(self.fun_dim + self.ref * self.ref, self.n_hidden * 2, self.n_hidden, n_layers=0, res=False, act=self.act).to(device)
        else:
            self.preprocess = MLP(self.fun_dim + self.space_dim, self.n_hidden * 2, self.n_hidden, n_layers=0, res=False, act=self.act).to(device)

        if self.Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden), nn.SiLU(), nn.Linear(self.n_hidden, self.n_hidden)).to(device)

        self.blocks = nn.ModuleList([Transolver_block(num_heads=self.n_head, hidden_dim=self.n_hidden,
                                                      dropout=self.dropout,
                                                      act=self.act,
                                                      mlp_ratio=self.mlp_ratio,
                                                      out_dim=self.out_dim * self.out_steps,
                                                      slice_num=self.slice_num,
                                                      H=self.H,
                                                      W=self.W,
                                                      last_layer=(_ == self.n_layers - 1))
                                     for _ in range(self.n_layers)]).to(device)
        self.initialize_weights()
        self.placeholder = nn.Parameter((1 / (self.n_hidden)) * torch.rand(self.n_hidden, dtype=torch.float)).to(device)

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_grid(self, batchsize=1):
        size_x, size_y = self.H, self.W
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        grid = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 2

        gridx = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 1, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = torch.cat((gridx, gridy), dim=-1).cuda()  # B H W 8 8 2

        pos = torch.sqrt(torch.sum((grid[:, :, :, None, None, :] - grid_ref[:, None, None, :, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, size_x, size_y, self.ref * self.ref).contiguous()
        return pos


    def forward(self, x: torch.Tensor, static_data: list) -> torch.Tensor:
        """
        Forward pass with autoregressive prediction.
        
        Args:
            x: (B, T*C, H, W) - Temporal snapshots concatenated along channel dimension
            static_data: List[Tensor] - Static features (e.g., coordinates)
        
        Returns:
            (B, seq_len, C, H, W) - Predicted sequence
        """
        B, C = x.shape[:2]
        spatial_shape = x.shape[2:]
        current_input = x.permute(0, 2, 3, 1).reshape(B, -1, C)
        x_pos = static_data[0]
        T = None

        predictions = []
        for t in range(self.seq_len):
            if self.unified_pos:
                x_feat = self.pos.repeat(x_pos.shape[0], 1, 1, 1).reshape(x_pos.shape[0], -1, self.ref * self.ref)

            if current_input is not None:
                fx = torch.cat((x_feat, current_input), -1)
                fx = self.preprocess(fx)
            else:
                fx = self.preprocess(x_feat)
                fx = fx + self.placeholder[None, None, :]

            if T is not None:
                Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x_feat.shape[1], 1)
                Time_emb = self.time_fc(Time_emb)
                fx = fx + Time_emb

            for block in self.blocks:
                fx = block(fx)
            
            # save prediction
            predictions.append(fx)
            current_input = torch.cat((current_input[..., self.out_dim * self.out_steps:], fx), dim=-1)

        # Reshape output to (B, seq_len, C, H, W)
        predictions = torch.stack(predictions, dim=1).permute(0, 1, 3, 2).view(B, -1, self.out_dim, *spatial_shape)

        return predictions

