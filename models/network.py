import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import *


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class Inp_ResBlock(nn.Module):

    def __init__(self, dim, n_res):
        super(Inp_ResBlock, self).__init__()
        self.patch_embed = OverlapPatchEmbed(3, dim)
        self.dim = dim

        level1 = [Resblock(dim=dim) for _ in range(n_res[0])]
        self.level1 = nn.Sequential(*level1)
        self.down1 = Downsample(dim)

        level2 = [Resblock(dim=int(dim*2**1)) for _ in range(n_res[1])]
        self.level2 = nn.Sequential(*level2)
        self.down2 = Downsample(int(dim*2**1))

        level3 = [Resblock(dim=int(dim * 2 ** 2)) for _ in range(n_res[2])]
        self.level3 = nn.Sequential(*level3)
        self.down3 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4

        self.prior = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 3) + int(dim * 2 ** 2) + int(dim * 2 ** 1), kernel_size=1)

    def forward(self, img):
        feats = []
        embed = self.patch_embed(img)

        lv1 = self.level1(embed)
        feats.append(lv1)
        lv1 = self.down1(lv1)

        lv2 = self.level2(lv1)
        feats.append(lv2)
        lv2 = self.down2(lv2)

        lv3 = self.level3(lv2)
        feats.append(lv3)
        lv3 = self.down3(lv3)

        prior = self.prior(lv3)
        P3, P2, P1 = torch.split(prior, (int(self.dim * 2 ** 3), int(self.dim * 2 ** 2), int(self.dim * 2 ** 1)), 1)
        return feats, [P3, P2, P1]


class PriorPrompt(nn.Module):

    def __init__(self, prompt_dim, LayerNorm_type, ffn_expansion_factor, prompt_len=5, prompt_size=16, in_dim=64, heads=4):
        super(PriorPrompt, self).__init__()
        self.prompt_size = prompt_size
        self.prompt_param = nn.Parameter(torch.randn(1, prompt_len, prompt_dim, prompt_size, prompt_size),
                                         requires_grad=True)
        self.conv_dw = nn.Sequential(nn.Conv2d(in_dim, prompt_dim, kernel_size=1),
                                     nn.GELU(),
                                     nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, groups=prompt_dim))

        self.conv_h = nn.Sequential(nn.Conv2d(prompt_dim, prompt_dim, kernel_size=(1, 21), padding=(0, 10), groups=prompt_dim),
                                    nn.Softmax(dim=-1))
        self.conv_w = nn.Sequential(nn.Conv2d(prompt_dim, prompt_dim, kernel_size=(21, 1), padding=(10, 0), groups=prompt_dim),
                                    nn.Softmax(dim=-2))

        self.conv_out = nn.Conv2d(prompt_dim, in_dim, kernel_size=3, padding=1)
        self.attn = TransformerBlock(dim=in_dim * 2, num_heads=heads, ffn_expansion_factor=ffn_expansion_factor,
                                     bias=False, LayerNorm_type=LayerNorm_type, only_CA=True)
        self.channel_reduction = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1)

    def forward(self, x, prior):
        B, C, H, W = x.shape

        u = self.conv_dw(prior)
        uh = F.interpolate(self.conv_h(u), (self.prompt_size, self.prompt_size), mode='bilinear').unsqueeze(1)
        uw = F.interpolate(self.conv_w(u), (self.prompt_size, self.prompt_size), mode='bilinear').unsqueeze(1)
        prompt = self.prompt_param.repeat(B, 1, 1, 1, 1)
        prompt = (uh * prompt) + (uw * prompt)

        prompt = torch.sum(prompt, dim=1)
        prompt = self.conv_out(prompt)
        prompt = F.interpolate(prompt, (H, W), mode='bilinear')
        out = torch.cat([x, prompt], dim=1)
        out = self.attn(out)
        out = self.channel_reduction(out)
        return out


class Model(nn.Module):

    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[1, 2, 2, 4],
                 num_refinement_blocks=1,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 dim_head=48,
                 offset_kernel=[9, 7, 5],
                 offset_stride=[8, 4, 2]):
        super(Model, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.inp_embed = Inp_ResBlock(dim=dim, n_res=[1, 2, 4])
        self.prompt1 = PriorPrompt(prompt_dim=64, prompt_len=5, prompt_size=64, in_dim=96, LayerNorm_type=LayerNorm_type, heads=heads[-1], ffn_expansion_factor=ffn_expansion_factor)
        self.prompt2 = PriorPrompt(prompt_dim=128, prompt_len=5, prompt_size=32, in_dim=192, LayerNorm_type=LayerNorm_type, heads=heads[-2], ffn_expansion_factor=ffn_expansion_factor)
        self.prompt3 = PriorPrompt(prompt_dim=320, prompt_len=5, prompt_size=16, in_dim=384, LayerNorm_type=LayerNorm_type, heads=heads[-3], ffn_expansion_factor=ffn_expansion_factor)

        encoder_level1 = [TransformerEncoder(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type,
                                             offset_kernel=offset_kernel[0], offset_stride=offset_stride[0], dim_head=dim_head)
                          for i in range(num_blocks[0])]
        self.encoder_level1 = nn.ModuleList(encoder_level1)

        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2

        encoder_level2 = [TransformerEncoder(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type,
                                             offset_kernel=offset_kernel[1], offset_stride=offset_stride[1], dim_head=dim_head)
                          for i in range(num_blocks[1])]
        self.encoder_level2 = nn.ModuleList(encoder_level2)

        self.down2_3 = Downsample(int(dim * 2 ** 1))  ## From Level 2 to Level 3

        encoder_level3 = [TransformerEncoder(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                             LayerNorm_type=LayerNorm_type,
                                             offset_kernel=offset_kernel[2], offset_stride=offset_stride[2], dim_head=dim_head)
                          for i in range(num_blocks[2])]
        self.encoder_level3 = nn.ModuleList(encoder_level3)

        self.down3_4 = Downsample(int(dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                                                       bias=bias, LayerNorm_type=LayerNorm_type,
                                                       offset_kernel=offset_kernel[2], offset_stride=offset_stride[2], dim_head=dim_head) for i in range(num_blocks[3])])
        self.up4_3 = Upsample(int(dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)

        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2],
                                                               ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                                                               LayerNorm_type=LayerNorm_type,
                                                               offset_kernel=offset_kernel[2], offset_stride=offset_stride[2], dim_head=dim_head) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             offset_kernel=offset_kernel[1], offset_stride=offset_stride[1], dim_head=dim_head) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             offset_kernel=offset_kernel[0], offset_stride=offset_stride[0], dim_head=dim_head) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            TransformerBlock(dim=int(dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                             bias=bias, LayerNorm_type=LayerNorm_type,
                             offset_kernel=offset_kernel[0], offset_stride=offset_stride[0], dim_head=dim_head) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):
        img_embed, prior = self.inp_embed(inp_img)
        inp_enc_level1 = self.patch_embed(inp_img)

        out_enc_level1 = inp_enc_level1
        for blk in self.encoder_level1:
            out_enc_level1 = blk(out_enc_level1, img_embed[0])
        inp_enc_level2 = self.down1_2(out_enc_level1)

        out_enc_level2 = inp_enc_level2
        for blk in self.encoder_level2:
            out_enc_level2 = blk(out_enc_level2, img_embed[1])
        inp_enc_level3 = self.down2_3(out_enc_level2)

        out_enc_level3 = inp_enc_level3
        for blk in self.encoder_level3:
            out_enc_level3 = blk(out_enc_level3, img_embed[2])
        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        latent = self.prompt3(x=latent, prior=prior[0])

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        out_dec_level3 = self.prompt2(x=out_dec_level3, prior=prior[1])

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        out_dec_level2 = self.prompt1(x=out_dec_level2, prior=prior[2])

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
        return out_dec_level1

