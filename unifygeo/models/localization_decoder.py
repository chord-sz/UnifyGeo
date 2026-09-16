"""Hierarchical detailed-feature matching and localization decoder."""

import collections

import torch
import torch.nn as nn
from torch.nn import functional as F

from .layers import LayerNorm, double_conv_bn, xavier_init

class PermuteChannels(nn.Module):
    def __init__(self, B, C, H, W):
        super().__init__()
        self.B = B
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return torch.permute(x, (self.B, self.C, self.H, self.W))

class Normalization(nn.Module):
    def __init__(self, p, dim):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )

class VigorLocalizationDecoder(nn.Module):
    def __init__(self, feature_norm=None, train_rerank=False, config=None):
        super().__init__()

        self.grd_feature_to_descriptor1 = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(768, 32, 1)),
                ('permute', PermuteChannels(0, 2, 3, 1)),
                ('conv2', nn.Conv2d(12, 1, 1)),
                ('flatten', nn.Flatten(start_dim=1))
        ]))

        self.grd_feature_to_descriptor2 = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(768, 16, 1)),
                ('permute', PermuteChannels(0, 2, 3, 1)),
                ('conv2', nn.Conv2d(12, 1, 1)),
                ('flatten', nn.Flatten(start_dim=1))
        ]))

        self.grd_feature_to_descriptor3 = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(768, 8, 1)),
                ('permute', PermuteChannels(0, 2, 3, 1)),
                ('conv2', nn.Conv2d(12, 1, 1)),
                ('flatten', nn.Flatten(start_dim=1))
        ]))

        self.grd_feature_to_descriptor4 = nn.Sequential(collections.OrderedDict([
                ('conv1', nn.Conv2d(768, 4, 1)),
                ('permute', PermuteChannels(0, 2, 3, 1)),
                ('conv2', nn.Conv2d(12, 1, 1)),
                ('flatten', nn.Flatten(start_dim=1))
        ]))

        self.sat_normalization = Normalization(2, 1)

        self.deconv4 = nn.ConvTranspose2d(769, 384, 2, 2)
        self.deconv3 = nn.ConvTranspose2d(385, 192, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(193, 96, 2, 2)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(97, 48, 2, 2),
                                     nn.BatchNorm2d(48, eps=1e-6),
                                     nn.ReLU(inplace=True),
                                     nn.ConvTranspose2d(48, 24, 2, 2),
                                     )

        if feature_norm is not None:
            print('-> Decoder BN: ON')
            self.conv4 = double_conv_bn(768, 384)
            self.conv3 = double_conv_bn(384, 192)
            self.conv2 = double_conv_bn(192, 96)
            self.conv1 = nn.Sequential(nn.Conv2d(24, 8, 3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(8, eps=1e-6),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(8, 1, 3, stride=1, padding=1))

            self.conv4_norm = nn.Identity()
            self.conv3_norm = nn.Identity()
            self.conv2_norm = nn.Identity()

        else:
            print('-> Decoder BN: OFF')
            self.conv4 = double_conv(768, 384)
            self.conv3 = double_conv(384, 192)
            self.conv2 = double_conv(192, 96)
            self.conv1 = nn.Sequential(nn.Conv2d(24, 8, 3, stride=1, padding=1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(8, 1, 3, stride=1, padding=1))

            self.conv4_norm = nn.Identity()
            self.conv3_norm = nn.Identity()
            self.conv2_norm = nn.Identity()

        self.match_pre_norm = config.decoder_match_norm
        if self.match_pre_norm:
            print('-> match pre norm: True')
            self.sat_matching_block_norm1 = LayerNorm(768, eps=1e-6, data_format="channels_first")
            self.sat_matching_block_norm2 = LayerNorm(384, eps=1e-6, data_format="channels_first")
            self.sat_matching_block_norm3 = LayerNorm(192, eps=1e-6, data_format="channels_first")
            self.sat_matching_block_norm4 = LayerNorm(96, eps=1e-6, data_format="channels_first")

            self.grd_descriptor_norm1 = nn.LayerNorm(768, eps=1e-6)
            self.grd_descriptor_norm2 = nn.LayerNorm(384, eps=1e-6)
            self.grd_descriptor_norm3 = nn.LayerNorm(192, eps=1e-6)
            self.grd_descriptor_norm4 = nn.LayerNorm(96, eps=1e-6)
        else:
            print('-> match pre norm: False')

        self.train_rerank = train_rerank
        self.init_weights_()

    def init_weights_(self, ):
        self.apply(xavier_init)

    def forward(self, out_grd, out_sat):
        grd_feature_volume = out_grd['grd_feature_volume']

        grd_descriptor1 = self.grd_feature_to_descriptor1(grd_feature_volume)
        grd_descriptor2 = self.grd_feature_to_descriptor2(grd_feature_volume)
        grd_descriptor3 = self.grd_feature_to_descriptor3(grd_feature_volume)
        grd_descriptor4 = self.grd_feature_to_descriptor4(grd_feature_volume)

        if self.match_pre_norm:
            grd_descriptor1 = self.grd_descriptor_norm1(grd_descriptor1)
            grd_descriptor2 = self.grd_descriptor_norm2(grd_descriptor2)
            grd_descriptor3 = self.grd_descriptor_norm3(grd_descriptor3)
            grd_descriptor4 = self.grd_descriptor_norm4(grd_descriptor4)

        grd_descriptor_map1 = grd_descriptor1.unsqueeze(2).unsqueeze(3).repeat(1, 1, 12, 12)
        grd_descriptor_map2 = grd_descriptor2.unsqueeze(2).unsqueeze(3).repeat(1, 1, 24, 24)
        grd_descriptor_map3 = grd_descriptor3.unsqueeze(2).unsqueeze(3).repeat(1, 1, 48, 48)
        grd_descriptor_map4 = grd_descriptor4.unsqueeze(2).unsqueeze(3).repeat(1, 1, 96, 96)

        multiscale_sat = out_sat['sat_feats_map']
        sat_feature_volume = multiscale_sat[0]
        sat_feature_block1 = multiscale_sat[1]
        sat_feature_block2 = multiscale_sat[2]
        sat_feature_block3 = multiscale_sat[3]

        grd_des_len = grd_descriptor1.size()[1]
        sat_des_len = sat_feature_volume.size()[1]
        assert grd_des_len == sat_des_len

        if self.match_pre_norm:
            sat_feature_volume_norm = self.sat_matching_block_norm1(sat_feature_volume)
            matching_score1 = torch.sum(
                (F.normalize(grd_descriptor_map1, p=2, dim=1) * F.normalize(sat_feature_volume_norm, p=2, dim=1)), dim=1,
                keepdim=True)
        else:

            matching_score1 = torch.sum((F.normalize(grd_descriptor_map1, p=2, dim=1) * F.normalize(sat_feature_volume, p=2, dim=1)), dim=1, keepdim=True)

        x = torch.cat([matching_score1, self.sat_normalization(sat_feature_volume)], dim=1)

        x = self.deconv4(x)
        x = torch.cat([x, sat_feature_block1], dim=1)
        x = self.conv4(x)
        x = self.conv4_norm(x)

        grd_des_len = grd_descriptor2.size()[1]
        sat_des_len = x.size()[1]
        assert grd_des_len == sat_des_len

        if self.match_pre_norm:
            x_norm = self.sat_matching_block_norm2(x)
            matching_score2 = torch.sum(
                (F.normalize(grd_descriptor_map2, p=2, dim=1) * F.normalize(x_norm, p=2, dim=1)), dim=1,
                keepdim=True)
        else:

            matching_score2 = torch.sum(
                (F.normalize(grd_descriptor_map2, p=2, dim=1) * F.normalize(x, p=2, dim=1)), dim=1, keepdim=True)

        x = torch.cat([matching_score2, self.sat_normalization(x)], dim=1)
        x = self.deconv3(x)
        x = torch.cat([x, sat_feature_block2], dim=1)
        x = self.conv3(x)
        x = self.conv3_norm(x)

        grd_des_len = grd_descriptor3.size()[1]
        sat_des_len = x.size()[1]
        assert grd_des_len == sat_des_len

        if self.match_pre_norm:
            x_norm = self.sat_matching_block_norm3(x)
            matching_score3 = torch.sum(
                (F.normalize(grd_descriptor_map3, p=2, dim=1) * F.normalize(x_norm, p=2, dim=1)), dim=1,
                keepdim=True)
        else:
            matching_score3 = torch.sum(
                (F.normalize(grd_descriptor_map3, p=2, dim=1) * F.normalize(x, p=2, dim=1)), dim=1, keepdim=True)

        x = torch.cat([matching_score3, self.sat_normalization(x)], dim=1)
        x = self.deconv2(x)
        x = torch.cat([x, sat_feature_block3], dim=1)
        x = self.conv2(x)
        x = self.conv2_norm(x)

        grd_des_len = grd_descriptor4.size()[1]
        sat_des_len = x.size()[1]
        assert grd_des_len == sat_des_len

        if self.match_pre_norm:
            x_norm = self.sat_matching_block_norm4(x)
            matching_score4 = torch.sum(
                (F.normalize(grd_descriptor_map4, p=2, dim=1) * F.normalize(x_norm, p=2, dim=1)), dim=1,
                keepdim=True)
        else:
            matching_score4 = torch.sum(
                (F.normalize(grd_descriptor_map4, p=2, dim=1) * F.normalize(x, p=2, dim=1)), dim=1, keepdim=True)

        x = torch.cat([matching_score4, self.sat_normalization(x)], dim=1)
        x = self.deconv1(x)
        x = self.conv1(x)

        logits_flattened = torch.flatten(x, start_dim=1)
        heatmap = torch.reshape(nn.Softmax(dim=-1)(logits_flattened), x.size())

        if not self.train_rerank:
            return logits_flattened, heatmap, matching_score1, matching_score2, matching_score3, matching_score4
        rerank_map = sat_feature_volume_norm if self.match_pre_norm else sat_feature_volume
        return logits_flattened, heatmap, matching_score1, matching_score2, matching_score3, matching_score4, grd_descriptor1, rerank_map

def build_ccvpe_decoder(args):
    return VigorLocalizationDecoder(
        feature_norm=args.decoder_norm,
        train_rerank=args.train_rerank,
        config=args,
    )
