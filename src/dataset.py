"""2D slice dataset over the preprocessed cases, with modality selection."""
import numpy as np
import pandas as pd
import torch
from monai.transforms import Compose, RandFlipd, RandRotated
from torch.utils.data import Dataset

from common import load_case


def build_augment(cfg):
    a = cfg["augment"]
    return Compose([
        # Axis 0 of a slice is the left-right axis of the BraTS volume
        RandFlipd(keys=["image", "label"], prob=a["flip_prob"], spatial_axis=0),
        RandRotated(keys=["image", "label"], range_x=np.deg2rad(a["rotate_deg"]), prob=1.0,
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
    ])


class SliceDataset(Dataset):
    def __init__(self, cfg, cases, channels, augment=False):
        self.cfg = cfg
        self.channels = channels
        self.transform = build_augment(cfg) if augment else None
        slices = pd.read_csv(cfg["paths"]["processed_dir"] / "slices.csv")
        slices = slices[slices["case"].isin(set(cases))]
        self.index = list(zip(slices["case"], slices["slice"]))
        self._cache = {}  # memmaps, opened lazily so each DataLoader worker has its own

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        case, s = self.index[i]
        if case not in self._cache:
            self._cache[case] = load_case(self.cfg, case)
        img, seg = self._cache[case]
        x = torch.from_numpy(img[s, self.channels].astype(np.float32))      # (C, H, W)
        y = torch.from_numpy(seg[s][None].astype(np.int64))                  # (1, H, W)
        if self.transform:
            d = self.transform({"image": x, "label": y})
            x, y = torch.as_tensor(d["image"]), torch.as_tensor(d["label"]).long()
        return x, y
