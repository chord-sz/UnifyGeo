"""Data-loading helpers for evaluation."""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Sampler


class PresetIndexSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def build_evaluation_transforms():
    normalize = A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    satellite = A.Compose([
        A.Resize(384, 384, interpolation=cv2.INTER_LINEAR_EXACT),
        normalize,
        ToTensorV2(),
    ])
    ground = A.Compose([
        A.Resize(384, 768, interpolation=cv2.INTER_LINEAR_EXACT),
        normalize,
        ToTensorV2(),
    ])
    return satellite, ground
