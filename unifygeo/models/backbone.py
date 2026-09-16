"""ConvNeXt feature encoder used by UnifyGeo."""

from functools import partial

import timm
import torch.nn as nn

from .layers import LayerNorm


class Backbone(nn.Module):
    def __init__(self, model_name, feature_norm=None):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2),
        )
        channels = self.model.feature_info.channels()
        self.with_norm = feature_norm is not None
        if self.with_norm:
            norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
            for index, channel_count in enumerate(channels):
                layer = norm_layer(channel_count)
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.weight, 1.0)
                self.add_module(f"norm{index}", layer)

    def get_config(self):
        return timm.data.resolve_model_data_config(self.model)

    def set_grad_checkpointing(self, enable=True):
        self.model.set_grad_checkpointing(enable)

    def forward(self, images):
        features = self.model(images)
        if not self.with_norm:
            return features
        return tuple(getattr(self, f"norm{index}")(feature) for index, feature in enumerate(features))


def build_backbone(config, feature_norm=None):
    return Backbone(config.model, feature_norm=feature_norm)
