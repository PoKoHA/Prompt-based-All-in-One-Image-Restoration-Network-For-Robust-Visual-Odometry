import torch
# print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

import einops
from einops import rearrange
from einops.layers.torch import Rearrange
import time

import matplotlib.pyplot as plt
##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class resblock(nn.Module):
    def __init__(self, dim):
        super(resblock, self).__init__()
        # self.norm = LayerNorm(dim, LayerNorm_type='BiasFree')

        self.body = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PReLU(),
                                  nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        res = self.body((x))
        res += x
        return res


class Chanel_Cross_Attention(nn.Module):
    def __init__(self, dim, num_head, bias):
        super(Chanel_Cross_Attention, self).__init__()
        self.num_head = num_head
        self.temperature = nn.Parameter(torch.ones(num_head, 1, 1), requires_grad=True)

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        # x -> q, y -> kv
        assert x.shape == y.shape, 'The shape of feature maps from image and features are not equal!'

        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_head)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_head)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_head, h=h, w=w)

        out = self.project_out(out)
        return out
##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

###################################################
# Deformable Cross Attention
###################################################
class LayerNormProxy(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')


class DCA(nn.Module):

    def __init__(self,
                 dim,
                 offset_kernel,
                 offset_stride,
                 dim_head,
                 heads,
                 LayerNorm_type):
        super(DCA, self).__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads

        self.scale = dim_head ** -0.5
        self.proj_q = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.proj_k = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.proj_v = nn.Conv2d(dim, inner_dim, kernel_size=1)

        self.conv_offset = nn.Sequential(nn.Conv2d(dim_head, dim_head, offset_kernel, stride=offset_stride,
                                                   padding=offset_kernel // 2, groups=dim_head),
                                         LayerNormProxy(dim_head),
                                         nn.GELU(),
                                         nn.Conv2d(dim_head, 2, 1, 1, 0, bias=False))
        self.proj_out = nn.Conv2d(inner_dim, dim, 1)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.heads, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x, kv):
        shape = x.shape
        B, C, H, W, head = *shape, self.heads
        dtype, device = x.dtype, x.device
        # visual_map("Query", kv)
        # visual_map("KV", kv)
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.heads, c=self.dim_head)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        pos = (offset + reference).clamp(-1., +1.)

        kv_sampled = F.grid_sample(input=kv.reshape(B * self.heads, self.dim_head, H, W),
                                   grid=pos[..., (1, 0)],  # y, x -> x, y
                                   mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        kv_sampled = kv_sampled.reshape(B, C, 1, n_sample)
        q = q.reshape(B * self.heads, self.dim_head, H * W)
        k = self.proj_k(kv_sampled).reshape(B * self.heads, self.dim_head, n_sample)
        v = self.proj_v(kv_sampled).reshape(B * self.heads, self.dim_head, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        y = self.proj_out(out)
        return y


class SelfDCA(nn.Module):

    def __init__(self,
                 dim,
                 offset_kernel,
                 offset_stride,
                 dim_head,
                 heads,
                 LayerNorm_type):
        super(SelfDCA, self).__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads

        self.scale = dim_head ** -0.5
        self.proj_q = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.proj_k = nn.Conv2d(dim, inner_dim, kernel_size=1)
        self.proj_v = nn.Conv2d(dim, inner_dim, kernel_size=1)

        self.conv_offset = nn.Sequential(nn.Conv2d(dim_head, dim_head, offset_kernel, stride=offset_stride,
                                                   padding=offset_kernel // 2, groups=dim_head),
                                         LayerNormProxy(dim_head),
                                         nn.GELU(),
                                         nn.Conv2d(dim_head, 2, 1, 1, 0, bias=False))
        self.proj_out = nn.Conv2d(inner_dim, dim, 1)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.heads, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, x):
        shape = x.shape
        B, C, H, W, head = *shape, self.heads
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=self.heads, c=self.dim_head)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        pos = (offset + reference).clamp(-1., +1.)
        kv_sampled = F.grid_sample(input=x.reshape(B * self.heads, self.dim_head, H, W),
                                   grid=pos[..., (1, 0)],  # y, x -> x, y
                                   mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        kv_sampled = kv_sampled.reshape(B, C, 1, n_sample)
        q = q.reshape(B * self.heads, self.dim_head, H * W)
        k = self.proj_k(kv_sampled).reshape(B * self.heads, self.dim_head, n_sample)
        v = self.proj_v(kv_sampled).reshape(B * self.heads, self.dim_head, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        y = self.proj_out(out)
        return y
##########################################################################
## Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 offset_kernel=3, offset_stride=1, dim_head=48, only_CA=False):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):

    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,
                 offset_kernel, offset_stride, dim_head):
        super(TransformerEncoder, self).__init__()
        self.norm1_1 = LayerNorm(dim, LayerNorm_type)
        self.norm1_2 = LayerNorm(dim, LayerNorm_type)
        self.spatial_attn = DCA(dim, offset_kernel=offset_kernel, offset_stride=offset_stride,
                                dim_head=dim_head, heads=num_heads, LayerNorm_type=LayerNorm_type)

        self.norm2_1 = LayerNorm(dim, LayerNorm_type)
        self.norm2_2 = LayerNorm(dim, LayerNorm_type)
        self.channel_attn = Chanel_Cross_Attention(dim=dim, num_head=num_heads, bias=bias)

        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y=None):
        x = x + self.spatial_attn(self.norm1_1(x), self.norm1_2(y))
        x = x + self.channel_attn(self.norm2_1(x), self.norm2_2(y))
        x = x + self.ffn(self.norm3(x))
        return x


class Resblock(nn.Module):

    def __init__(self, dim):
        super(Resblock, self).__init__()
        self.norm0 = nn.BatchNorm2d(dim)
        self.relu0 = nn.PReLU(dim)
        self.conv0 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.PReLU(dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv0(self.relu0(self.norm0(x)))
        out = self.conv1(self.relu1(self.norm1(out)))
        return out + x


def visual_map(title, map):
    map = map.squeeze(0)
    gray = torch.sum(map, 0)
    gray = gray / map.shape[0]
    gray = gray.detach().cpu().numpy()

    plt.imshow(gray)
    plt.title(title)
    plt.show()
