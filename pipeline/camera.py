# -*- coding: utf-8 -*-
"""针孔相机模型：把照片上的像素坐标换算为观测方向单位向量。

约定：
    - 光轴沿 +z 方向，主点取图像中心；
    - 图像 x 轴向右、y 轴向下；
    - 该坐标系与 ICRS 天球坐标系同为右手系，两者之间只差一个旋转，
      可由后续 Kabsch 对齐直接求解，因此无需手动翻转 y 轴。
"""
from __future__ import annotations

import numpy as np


def focal_length_px(f_mm: float, pixel_size_mm: float) -> float:
    """焦距换算为像素单位：f_px = f_mm / pixel_size_mm。"""
    return f_mm / pixel_size_mm


def pixels_to_unit_vectors(pixels, width: int, height: int, f_px: float) -> np.ndarray:
    """把像素坐标批量转换为单位方向向量。

    参数:
        pixels: shape (N,2) 或 [(x, y), ...]
        width, height: 图像宽高（像素）
        f_px: 焦距（像素）

    返回:
        ndarray，shape (N,3)，单位方向向量，z 分量恒为正（光轴朝前）。
    """
    pts = np.asarray(pixels, dtype=float).reshape(-1, 2)
    cx, cy = width / 2.0, height / 2.0

    # 针孔投影的逆变换：归一化坐标 + 光轴方向
    xn = (pts[:, 0] - cx) / f_px
    yn = (pts[:, 1] - cy) / f_px
    dirs = np.column_stack([xn, yn, np.ones(len(pts))])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    return dirs
