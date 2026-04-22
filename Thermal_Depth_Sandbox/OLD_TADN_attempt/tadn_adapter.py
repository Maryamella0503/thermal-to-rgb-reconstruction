from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2


@dataclass
class TADNConfig:
    clip_percentiles: Tuple[float, float] = (2.0, 98.0)

    use_bilateral: bool = True
    bilateral_d: int = 7
    bilateral_sigma_color: float = 40.0
    bilateral_sigma_space: float = 40.0

    use_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    use_edge_boost: bool = True
    edge_weight: float = 0.35
    sobel_ksize: int = 3
    edge_blur_ksize: int = 3

    use_gamma: bool = True
    gamma: float = 0.9

    out_size: Optional[Tuple[int, int]] = None  # (W,H)


class TADNAdapter:
    """
    Input:  depth map as float numpy array (H,W) or (H,W,1)
    Output: uint8 RGB depth image (H,W,3) ready for ControlNet depth conditioning
    """

    def __init__(self, cfg: Optional[TADNConfig] = None):
        self.cfg = cfg or TADNConfig()

    def _to_2d_float(self, depth: np.ndarray) -> np.ndarray:
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"Expected depth shape (H,W) or (H,W,1), got {depth.shape}")
        depth = depth.astype(np.float32)
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth

    def _robust_minmax(self, depth: np.ndarray) -> np.ndarray:
        lo_p, hi_p = self.cfg.clip_percentiles
        lo = np.percentile(depth, lo_p)
        hi = np.percentile(depth, hi_p)
        if hi <= lo + 1e-6:
            lo, hi = float(depth.min()), float(depth.max()) + 1e-6
        depth = np.clip(depth, lo, hi)
        depth = (depth - lo) / (hi - lo + 1e-8)
        return depth

    def _apply_gamma(self, x01: np.ndarray) -> np.ndarray:
        if not self.cfg.use_gamma:
            return x01
        g = float(self.cfg.gamma)
        g = max(0.05, min(g, 5.0))
        return np.power(np.clip(x01, 0.0, 1.0), g)

    def _clahe(self, x01: np.ndarray) -> np.ndarray:
        if not self.cfg.use_clahe:
            return x01
        x8 = (np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.cfg.clahe_clip_limit),
            tileGridSize=tuple(self.cfg.clahe_tile_grid_size),
        )
        x8 = clahe.apply(x8)
        return x8.astype(np.float32) / 255.0

    def _denoise(self, x01: np.ndarray) -> np.ndarray:
        if not self.cfg.use_bilateral:
            return x01
        x8 = (np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)
        x8 = cv2.bilateralFilter(
            x8,
            d=int(self.cfg.bilateral_d),
            sigmaColor=float(self.cfg.bilateral_sigma_color),
            sigmaSpace=float(self.cfg.bilateral_sigma_space),
        )
        return x8.astype(np.float32) / 255.0

    def _edge_boost(self, x01: np.ndarray) -> np.ndarray:
        if not self.cfg.use_edge_boost:
            return x01

        x8 = (np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)

        k = int(self.cfg.sobel_ksize)
        gx = cv2.Sobel(x8, cv2.CV_32F, 1, 0, ksize=k)
        gy = cv2.Sobel(x8, cv2.CV_32F, 0, 1, ksize=k)
        mag = cv2.magnitude(gx, gy)

        mag = mag / (mag.max() + 1e-8)
        if self.cfg.edge_blur_ksize and self.cfg.edge_blur_ksize > 1:
            mag = cv2.GaussianBlur(mag, (self.cfg.edge_blur_ksize, self.cfg.edge_blur_ksize), 0)

        w = float(self.cfg.edge_weight)
        boosted = np.clip(x01 + w * mag, 0.0, 1.0)
        return boosted

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        d = self._to_2d_float(depth)
        d01 = self._robust_minmax(d)
        d01 = self._apply_gamma(d01)
        d01 = self._denoise(d01)
        d01 = self._clahe(d01)
        d01 = self._edge_boost(d01)

        out8 = (np.clip(d01, 0.0, 1.0) * 255.0).astype(np.uint8)

        if self.cfg.out_size is not None:
            w, h = self.cfg.out_size
            out8 = cv2.resize(out8, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

        out_rgb = cv2.cvtColor(out8, cv2.COLOR_GRAY2RGB)
        return out_rgb