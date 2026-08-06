# -*- coding: utf-8 -*-
"""图像星点检测。

思路（与原项目一致）：
    1. 转灰度；
    2. 固定阈值二值化，只保留接近饱和的亮斑（星点核心）；
    3. 连通域分析，取每个亮斑的质心作为星点坐标；
    4. 剔除彼此过近的重复点。

改进点：
    - 按亮斑面积降序处理，优先保留更亮的星，结果顺序确定且可复现；
    - 调试中间图可开关，默认不落盘；
    - 读取失败时给出明确报错。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DetectionResult:
    """星点检测结果。"""

    points: list        # [(x, y), ...]，按“亮度”降序
    width: int          # 图像宽度（像素）
    height: int         # 图像高度（像素）


def detect_star_points(image_path, threshold=252, min_distance=10,
                       save_debug=False, debug_dir=None) -> DetectionResult:
    """从照片中检测星点。

    参数:
        image_path: 图像文件路径
        threshold: 亮区二值化阈值（0-255），越大要求越亮、检测到的星越少
        min_distance: 两个星点之间的最小像素间距
        save_debug: 是否保存调试中间图
        debug_dir: 调试图输出目录（save_debug=True 时生效）

    返回:
        DetectionResult
    """
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]

    # 固定阈值二值化：只保留接近饱和的星点核心
    _, bright = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    if save_debug:
        _save_debug_images(gray, bright, debug_dir or DEBUG_FALLBACK)

    # 连通域分析
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积降序：面积大的亮斑通常更亮，优先保留（也决定“最亮前 N 颗”的选取）
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    points = []
    for cnt in contours:
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        if _too_close(cx, cy, points, min_distance):
            continue
        points.append((cx, cy))

    return DetectionResult(points=points, width=width, height=height)


DEBUG_FALLBACK = Path("debug")


def _too_close(x, y, existing, min_distance) -> bool:
    """判断 (x, y) 是否与 existing 中某个点距离小于 min_distance。"""
    existing = np.asarray(existing, dtype=float)
    if existing.size == 0:
        return False
    dist = np.hypot(existing[:, 0] - x, existing[:, 1] - y)
    return bool(np.any(dist < min_distance))


def _save_debug_images(gray, bright, debug_dir) -> None:
    """保存预处理中间图，便于调试阈值与形态学参数（原项目固定写入 Trash/）。"""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imwrite(str(debug_dir / "1_gray.jpg"), gray)
    cv2.imwrite(str(debug_dir / "2_blurred.jpg"), blurred)
    cv2.imwrite(str(debug_dir / "6_bright_regions.jpg"), bright)

    # 自适应阈值 + 闭运算：原项目探索过，当前流程不使用，仅作对比保留
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 17, 11,
    )
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cv2.imwrite(str(debug_dir / "3_thresh.jpg"), thresh)
    cv2.imwrite(str(debug_dir / "4_closed.jpg"), closed)
    cv2.imwrite(str(debug_dir / "5_closed_inv.jpg"), 255 - closed)
