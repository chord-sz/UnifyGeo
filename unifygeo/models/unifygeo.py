"""UnifyGeo model definition for VIGOR evaluation."""

import numpy as np
import torch
import torch.nn as nn

from .backbone import build_backbone
from .feature_decoder import build_feature_decoder
from .localization_decoder import build_ccvpe_decoder


class UnifyGeo(nn.Module):
    def __init__(
        self,
        grd_encoder,
        grd_global_feats_decoder,
        grd_local_feats_decoder,
        sat_encoder,
        sat_global_feats_decoder,
        sat_local_feats_decoder,
        pyramid_decoder,
        config,
    ):
        super().__init__()
        self.grd_encoder = grd_encoder
        self.grd_global_feats_decoder = grd_global_feats_decoder
        self.grd_local_feats_decoder = grd_local_feats_decoder
        self.sat_encoder = sat_encoder
        self.sat_global_feats_decoder = sat_global_feats_decoder
        self.sat_local_feats_decoder = sat_local_feats_decoder
        self.pyramid_decoder = pyramid_decoder
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_rerank = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.aux_loss = config.aux_loss
        self.train_rerank = config.train_rerank

    def get_config(self):
        return self.grd_encoder.get_config()

    def set_grad_checkpointing(self, enable=True):
        self.grd_encoder.set_grad_checkpointing(enable)
        self.sat_encoder.set_grad_checkpointing(enable)

    def forward_features_grd(self, images):
        pyramid = self.grd_encoder(images)
        return {
            "grd_global_feats": self.grd_global_feats_decoder(pyramid[-1]),
            "grd_feature_volume": self.grd_local_feats_decoder(pyramid[-1]),
        }

    def forward_features_sat(self, images):
        pyramid = list(self.sat_encoder(images))
        local_features = self.sat_local_feats_decoder(pyramid[-1])
        pyramid.append(local_features)
        return {
            "sat_global_feats": self.sat_global_feats_decoder(pyramid[-2]),
            "sat_feats_map": pyramid[::-1],
        }

    def forward_decoder(self, out_grd, out_sat):
        decoder_outputs = self.pyramid_decoder(out_grd, out_sat)
        logits, heatmap, score1, score2, score3, score4 = decoder_outputs[:6]
        out_grd["match_attns"] = score1
        out_grd["cls_logits"] = logits
        out_grd["heatmap"] = heatmap
        if self.aux_loss:
            out_grd["aux_outputs"] = [
                {"match_attns": score2},
                {"match_attns": score3},
                {"match_attns": score4},
            ]
        if self.train_rerank:
            out_grd["rerank_grd_desc"] = decoder_outputs[6]
            out_sat["rerank_sat_map"] = decoder_outputs[7]
        return out_grd, out_sat

    def forward(self, image1, image2=None, expansion_num=None):
        if image2 is not None:
            if expansion_num is not None:
                image1 = image1.repeat_interleave(expansion_num, dim=0)
            out_grd = self.forward_features_grd(image1)
            out_sat = self.forward_features_sat(image2)
            return self.forward_decoder(out_grd, out_sat)
        if image1.shape[-1] != image1.shape[-2]:
            return self.forward_features_grd(image1)
        return self.forward_features_sat(image1)


def build_model(config):
    grd_encoder = build_backbone(config, feature_norm=config.backbone_norm)
    grd_global = build_feature_decoder(config, aggregator_type=config.grd_aggregator)
    grd_local = build_feature_decoder(config, aggregator_type=None)
    sat_encoder = build_backbone(config, feature_norm=config.backbone_norm)
    sat_global = build_feature_decoder(config, aggregator_type=config.sat_aggregator)
    sat_local = build_feature_decoder(config, aggregator_type=None)
    localization_decoder = build_ccvpe_decoder(config)
    return UnifyGeo(
        grd_encoder,
        grd_global,
        grd_local,
        sat_encoder,
        sat_global,
        sat_local,
        localization_decoder,
        config,
    )
