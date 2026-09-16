"""Training losses retained for the future training-code release."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalInfoNCE(nn.Module):
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, ground_features, aerial_features, logit_scale):
        ground_features = F.normalize(ground_features, dim=-1)
        aerial_features = F.normalize(aerial_features, dim=-1)
        logits = logit_scale * ground_features @ aerial_features.T
        labels = torch.arange(len(logits), device=logits.device)
        return (self.cross_entropy(logits, labels) + self.cross_entropy(logits.T, labels)) / 2


def match_infonce(scores, labels, temperature=0.1):
    exp_scores = torch.exp(scores / temperature)
    positive_mask = labels > 1e-2
    probabilities = exp_scores / torch.sum(exp_scores, dim=1, keepdim=True)
    positive_log_probabilities = torch.log(torch.masked_select(probabilities, positive_mask))
    positive_weights = torch.masked_select(labels, positive_mask)
    return -torch.sum(positive_log_probabilities * positive_weights) / torch.sum(positive_weights)


def distribution_cross_entropy(logits, labels):
    return -torch.sum(labels * nn.LogSoftmax(dim=1)(logits)) / logits.size(0)
