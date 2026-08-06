# -*- coding: utf-8 -*-
"""几何函数的最小自检：直接运行 python tests/test_geometry.py。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from pipeline import geometry


def test_radec_conversion():
    # 赤经 0°、赤纬 0° -> (1, 0, 0)
    v = geometry.radec_to_unit_vectors([0.0], [0.0])
    np.testing.assert_allclose(v[0], [1, 0, 0], atol=1e-12)
    # 北极（赤纬 90°）-> (0, 0, 1)
    v = geometry.radec_to_unit_vectors([123.0], [90.0])
    np.testing.assert_allclose(v[0], [0, 0, 1], atol=1e-12)
    # 任意坐标模长恒为 1
    ras = np.linspace(0, 360, 50, endpoint=False)
    decs = np.linspace(-90, 90, 50)
    norms = np.linalg.norm(geometry.radec_to_unit_vectors(ras, decs), axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_angular_distance():
    a = np.array([[1, 0, 0]])
    b = np.array([[0, 1, 0]])
    np.testing.assert_allclose(geometry.angular_distance_deg(a, b), [90.0], atol=1e-10)
    np.testing.assert_allclose(geometry.angular_distance_deg(a, a), [0.0], atol=1e-10)


def test_pairwise_matrix():
    v = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    d = geometry.pairwise_angular_distances_deg(v)
    np.testing.assert_allclose(
        d, [[0, 90, 90], [90, 0, 90], [90, 90, 0]], atol=1e-10,
    )


def test_kabsch_rotation():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(10, 3))
    src /= np.linalg.norm(src, axis=1, keepdims=True)

    theta = np.radians(37.0)
    c, s = np.cos(theta), np.sin(theta)
    r_true = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    dst = src @ r_true.T

    r, residual = geometry.kabsch_rotation(src, dst)
    np.testing.assert_allclose(r, r_true, atol=1e-8)
    np.testing.assert_allclose(residual, 0.0, atol=1e-5)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n全部 {len(tests)} 项测试通过。")


if __name__ == "__main__":
    _run_all()
