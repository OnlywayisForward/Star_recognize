# -*- coding: utf-8 -*-
"""几何与坐标变换。

赤经赤纬（ICRS，度）<-> 天球单位向量、角距离、Kabsch 最优旋转对齐。
所有函数均使用 NumPy 向量化，避免逐颗星循环。
"""
from __future__ import annotations

import numpy as np


def radec_to_unit_vectors(ra_deg, dec_deg) -> np.ndarray:
    """把 ICRS 赤经赤纬批量转换为单位方向向量。

    参数:
        ra_deg, dec_deg: 标量或等长数组，单位：度

    返回:
        ndarray，shape (..., 3)，每一行是单位向量 (x, y, z)。
        约定：x = cos(dec)cos(ra)，y = cos(dec)sin(ra)，z = sin(dec)。
    """
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.stack([x, y, z], axis=-1)


def angular_distance_deg(u, v) -> np.ndarray:
    """计算两组向量（或两个向量）之间的夹角，单位：度。

    参数:
        u, v: shape 为 (..., 3) 的单位向量

    返回:
        夹角（度），形状与输入广播结果一致
    """
    dot = np.clip(np.einsum("...i,...i->...", u, v), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def pairwise_angular_distances_deg(vectors) -> np.ndarray:
    """计算 (N,3) 单位向量两两之间的角距离矩阵。

    返回:
        ndarray，shape (N, N)，对称矩阵，对角为 0。
        预计算一次后，匹配阶段枚举三角形时直接查表，避免重复反余弦运算。
    """
    vectors = np.asarray(vectors, dtype=float)
    dots = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def kabsch_rotation(src, dst):
    """Kabsch 算法：求最优旋转 R，使 src 旋转后与 dst 最小二乘对齐。

    参数:
        src, dst: shape (k, 3) 的对应点（至少 3 组），例如
                  src = 照片星单位向量，dst = 星表星单位向量

    返回:
        R: (3,3) 旋转矩阵，满足 src @ R.T ≈ dst
        residual_deg: (k,) 每个点的旋转后残差角度（度）
    """
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    h = src.T @ dst
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    # 保证是纯旋转（行列式为 +1）而不是镜像
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T

    rotated = src @ r.T
    residual = angular_distance_deg(rotated, dst)
    return r, residual
