"""VIGOR evaluation dataset for same-area and cross-area protocols."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VigorDatasetEval(Dataset):
    def __init__(self, data_root, protocol, image_type, transform=None):
        super().__init__()
        if protocol not in {"same-area", "cross-area"}:
            raise ValueError(f"Unsupported VIGOR protocol: {protocol}")
        if image_type not in {"reference", "query"}:
            raise ValueError(f"Unsupported image type: {image_type}")
        self.data_root = Path(data_root)
        self.protocol = protocol
        self.image_type = image_type
        self.transform = transform
        self.cities = (
            ["Chicago", "NewYork", "SanFrancisco", "Seattle"]
            if protocol == "same-area"
            else ["Chicago", "SanFrancisco"]
        )
        satellite_frames = []
        for city in self.cities:
            split_dir = self.data_root / "splits_new" / city
            satellite_file = split_dir / "satellite_list.txt"
            self._require_file(satellite_file)
            frame = pd.read_csv(satellite_file, header=None, sep=r"\s+")
            frame = frame.rename(columns={0: "satellite"})
            frame["path"] = frame["satellite"].map(
                lambda name: str(self.data_root / city / "satellite" / name)
            )
            satellite_frames.append(frame)
        satellite_table = pd.concat(satellite_frames, ignore_index=True)
        satellite_to_index = dict(zip(satellite_table.satellite, satellite_table.index))

        ground_frames = []
        for city in self.cities:
            split_dir = self.data_root / "splits_new" / city
            label_file = (
                split_dir / "same_area_balanced_test.txt"
                if protocol == "same-area"
                else split_dir / "pano_label_balanced.txt"
            )
            self._require_file(label_file)
            frame = pd.read_csv(label_file, header=None, sep=r"\s+")
            frame = frame.iloc[:, :13].rename(columns={
                0: "ground",
                1: "satellite",
                2: "delta_row",
                3: "delta_col",
                4: "semi1",
                5: "semi1_delta_row",
                6: "semi1_delta_col",
                7: "semi2",
                8: "semi2_delta_row",
                9: "semi2_delta_col",
                10: "semi3",
                11: "semi3_delta_row",
                12: "semi3_delta_col",
            })
            frame["path"] = frame["ground"].map(
                lambda name: str(self.data_root / city / "panorama" / name)
            )
            for column in ("satellite", "semi1", "semi2", "semi3"):
                frame[column] = frame[column].map(satellite_to_index)
            ground_frames.append(frame)
        ground_table = pd.concat(ground_frames, ignore_index=True)

        if image_type == "reference":
            self.images = satellite_table["path"].to_numpy()
            self.labels = satellite_table.index.to_numpy()
            self.deltas = None
        else:
            self.images = ground_table["path"].to_numpy()
            self.labels = ground_table[["satellite", "semi1", "semi2", "semi3"]].to_numpy()
            self.deltas = ground_table[["delta_row", "delta_col"]].to_numpy()

    @staticmethod
    def _require_file(path):
        if not path.is_file():
            raise FileNotFoundError(f"Required VIGOR split file is missing: {path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Unable to read VIGOR image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        label = torch.as_tensor(self.labels[index], dtype=torch.long)
        if self.image_type == "reference":
            return image, label

        row_offset, col_offset = self.deltas[index]
        row_offset = np.round(row_offset / 640 * 384)
        col_offset = np.round(col_offset / 640 * 384)
        x, y = np.meshgrid(
            np.linspace(-192 + col_offset, 192 + col_offset, 384),
            np.linspace(-192 - row_offset, 192 - row_offset, 384),
        )
        distance = np.sqrt(x * x + y * y)
        target_map = np.exp(-(distance ** 2) / (2.0 * 4.0 ** 2))[None].astype(np.float32)
        city = next((name for name in self.cities if name in image_path), None)
        if city is None:
            raise RuntimeError(f"Unable to infer the VIGOR city from: {image_path}")
        targets = {
            "src_location": torch.tensor([row_offset, col_offset]),
            "src_scores": torch.from_numpy(target_map),
            "city": city,
        }
        return image, label, targets
