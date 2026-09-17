"""Feature aggregation layers used by the retrieval branches.

SOABlock adapts SOLAR's solar_global/networks/networks.py::SOABlock (MIT):
https://github.com/tonyngjichun/SOLAR/blob/master/solar_global/networks/networks.py
This version supports configurable normalization, omits visualization outputs,
and adds optional gradient checkpointing for UnifyGeo's ConvNeXt features.
See THIRD_PARTY_NOTICES for the upstream copyright and license notice.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import LayerNorm


class GeM(nn.Module):
    """Generalized mean pooling with a learnable exponent."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        pooled = F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            (x.size(-2), x.size(-1)),
        )
        return pooled.pow(1.0 / self.p).flatten(1)


class SOABlock(nn.Module):
    """Second-order attention block adapted from SOLAR's global SOABlock."""

    def __init__(self, in_channels, reduction=2, norm="LN"):
        super().__init__()
        mid_channels = in_channels // reduction
        if norm == "BN":
            norm_layer = partial(nn.BatchNorm2d, eps=1e-6)
        elif norm == "LN":
            norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        else:
            norm_layer = nn.Identity
        self.mid_channels = mid_channels
        self.f = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1), norm_layer(mid_channels), nn.ReLU())
        self.g = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 1), norm_layer(mid_channels), nn.ReLU())
        self.h = nn.Conv2d(in_channels, mid_channels, 1)
        self.v = nn.Conv2d(mid_channels, in_channels, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.grad_checkpointing = False
        self.use_reentrant = False
        for layer in (self.f, self.g, self.h):
            layer.apply(self._init_projection)
        self.v.apply(self._init_zero)

    @staticmethod
    def _init_projection(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight.data)
            nn.init.constant_(module.bias.data, 0.0)

    @staticmethod
    def _init_zero(module):
        if isinstance(module, nn.Conv2d):
            nn.init.constant_(module.weight.data, 0.0)
            nn.init.constant_(module.bias.data, 0.0)

    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = bool(enable)

    def _forward_impl(self, x):
        batch, _, height, width = x.shape
        query = self.f(x).view(batch, self.mid_channels, height * width)
        key = self.g(x).view(batch, self.mid_channels, height * width)
        value = self.h(x).view(batch, self.mid_channels, height * width)
        attention = self.softmax((self.mid_channels ** -0.5) * torch.bmm(query.permute(0, 2, 1), key))
        output = torch.bmm(attention, value.permute(0, 2, 1))
        output = output.permute(0, 2, 1).view(batch, self.mid_channels, height, width)
        return self.v(output) + x

    def forward(self, x):
        if self.training and self.grad_checkpointing:
            return checkpoint(self._forward_impl, x, use_reentrant=self.use_reentrant)
        return self._forward_impl(x)


class SOAGeM(nn.Module):
    def __init__(self, channels=768, reduction=2, norm="LN"):
        super().__init__()
        self.block = SOABlock(channels, reduction, norm)
        self.gem = GeM()

    def set_grad_checkpointing(self, enable=True):
        self.block.set_grad_checkpointing(enable)

    def forward(self, x):
        return self.gem(self.block(x))
