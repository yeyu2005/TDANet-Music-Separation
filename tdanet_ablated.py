"""
Ablated TDANet Architecture (No Attention)

This code is built upon the official implementation of TDANet 
(Li, Yang & Hu, ICLR 2023). Original repo: https://github.com/JusperLee/TDANet

Modifications for this project:
- Surgically removed the Top-Down Attention mechanism (Global and Local) 
  to conduct an architectural ablation study.
- Adapted the output head for 4-stem music source separation.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.base_model import BaseModel

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class _LayerNorm(nn.Module):
    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    def forward(self, x):
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d, padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d, padding=((kSize - 1) // 2) * d, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class FFN(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = ConvNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv1d(hidden_size, hidden_size, 5, 1, 2, bias=True, groups=hidden_size)
        self.act = nn.ReLU()
        self.fc2 = ConvNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, in_channels, max_length):
        pe = torch.zeros(max_length, in_channels)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, in_channels, 2, dtype=torch.float) * -(math.log(10000.0) / in_channels)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super().__init__()
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, n_head, dropout, is_casual):
        super().__init__()
        self.pos_enc = PositionalEncoding(in_channels, 10000)
        self.attn_in_norm = nn.LayerNorm(in_channels)
        self.attn = nn.MultiheadAttention(in_channels, n_head, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
        self.is_casual = is_casual

    def forward(self, x):
        x = x.transpose(1, 2)
        attns = None
        output = self.pos_enc(self.attn_in_norm(x))
        output, _ = self.attn(output, output, output)
        output = self.norm(output + self.dropout(output))
        return output.transpose(1, 2)


# Ablated Global Attention: pass-through
class GA(nn.Module):
    def __init__(self, in_chan, out_chan, drop_path) -> None:
        super().__init__()

    def forward(self, x):
        return x


# Ablated Local Attention: local-only convolutional fusion
class LA(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)

    def forward(self, x_l, x_g):
        return self.local_embedding(x_l)


class UConvBlock(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1))

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=2 * stride + 1, stride=stride, groups=in_channels, d=1))

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(LA(in_channels, in_channels))

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.globalatt = nn.Identity()
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(LA(in_channels, in_channels, 5))

    def forward(self, x):
        residual = x.clone()
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        global_f = torch.zeros(output[-1].shape, requires_grad=False, device=output1.device)
        x_fused = []
        for idx in range(self.depth):
            local = output[idx]
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)

        return self.res_conv(expanded) + residual


class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, _iter=4):
        super().__init__()
        self.unet = UConvBlock(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        self.concat_block = nn.Sequential(nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU())

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(mixture + x))
        return x


class TDANetAblated(BaseModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=4,
        num_sources=2,
        sample_rate=16000,
    ):
        super(TDANetAblated, self).__init__(sample_rate=sample_rate)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.enc_num_basis = self.enc_kernel_size // 2 + 1
        self.num_sources = num_sources

        self.lcm = abs(self.enc_kernel_size // 4 * 4 ** self.upsampling_depth) // math.gcd(self.enc_kernel_size // 4, 4 ** self.upsampling_depth)

        self.encoder = nn.Conv1d(in_channels=1, out_channels=self.enc_num_basis, kernel_size=self.enc_kernel_size, stride=self.enc_kernel_size // 4, padding=self.enc_kernel_size // 2, bias=False)
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        self.ln = GlobLN(self.enc_num_basis)
        self.bottleneck = nn.Conv1d(in_channels=self.enc_num_basis, out_channels=out_channels, kernel_size=1)

        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks)

        mask_conv = nn.Conv1d(out_channels, num_sources * self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        self.decoder = nn.ConvTranspose1d(in_channels=self.enc_num_basis * num_sources, out_channels=num_sources, kernel_size=self.enc_kernel_size, stride=self.enc_kernel_size // 4, padding=self.enc_kernel_size // 2, groups=1, bias=False)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    def pad_input(self, input, window, stride):
        batch_size, nsample = input.shape
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)
        return input, rest

    def forward(self, input_wav):
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav
        if input_wav.ndim == 3:
            input_wav = input_wav.squeeze(1)

        x, rest = self.pad_input(input_wav, self.enc_kernel_size, self.enc_kernel_size // 4)
        x = self.encoder(x.unsqueeze(1))

        s = x.clone()
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)

        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = estimated_waveforms[:, : , self.enc_kernel_size - self.enc_kernel_size // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4), ].contiguous()
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def get_model_args(self):
        model_args = {"num_sources": self.num_sources, "sample_rate": self._sample_rate}
        return model_args
