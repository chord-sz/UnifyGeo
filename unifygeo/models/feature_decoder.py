"""Task-specific ConvNeXt stage and retrieval aggregation heads."""

import collections
import copy

import timm
import torch.nn as nn

from .aggregators import SOAGeM


class FeatureDecoder(nn.Module):
    def __init__(self, config, aggregator_type):
        super().__init__()
        backbone = timm.create_model(config.model, pretrained=False, features_only=True)
        self.model = nn.Sequential(collections.OrderedDict([
            ("downsample", copy.deepcopy(backbone.stages_3.downsample)),
            ("conv", copy.deepcopy(backbone.stages_3.blocks[0])),
        ]))
        if aggregator_type == "soa_gem":
            self.aggregator = SOAGeM(config.enc_dims, reduction=2, norm=config.aggregator_norm)
            self.agg_post_norm = nn.LayerNorm(config.enc_dims, eps=1e-6)
        elif aggregator_type is None:
            self.aggregator = nn.Identity()
            self.agg_post_norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported aggregator: {aggregator_type}")

    def forward(self, features):
        features = self.model(features)
        features = self.aggregator(features)
        return self.agg_post_norm(features)


def build_feature_decoder(config, aggregator_type):
    return FeatureDecoder(config, aggregator_type)
