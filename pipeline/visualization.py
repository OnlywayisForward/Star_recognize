# -*- coding: utf-8 -*-
"""结果可视化：在原图上标注星点与 HIP 编号。"""
from __future__ import annotations

from pathlib import Path

import cv2


def _load_image(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{image_path}")
    return img


def draw_star_points(image_path, star_points, output_path,
                     radius=5, color=(0, 255, 0)) -> None:
    """把所有检测到的星点画在原始照片上。"""
    img = _load_image(image_path)
    for x, y in star_points:
        cv2.circle(img, (int(x), int(y)), radius, color, -1)
    _write(img, output_path)


def draw_hip_labels(image_path, star_points, photo_to_hip, output_path) -> None:
    """在照片星点旁标注识别出的 HIP 编号。"""
    img = _load_image(image_path)
    for photo_idx, hip in photo_to_hip.items():
        if not 0 <= photo_idx < len(star_points):
            continue
        x, y = star_points[photo_idx]
        pos = (int(x), int(y))
        cv2.circle(img, pos, 5, (0, 255, 0), -1)
        cv2.putText(img, str(hip), (pos[0] + 5, pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    _write(img, output_path)


def _write(img, output_path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    print(f"      已保存：{output_path}")
