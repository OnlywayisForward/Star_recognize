# -*- coding: utf-8 -*-
"""星对哈希库。

把“标准星图中任意两颗星的角距离”按分桶建立索引：
    桶号 = round(角距离 / 桶宽)
    桶号 -> [(HIP_a, HIP_b), (HIP_b, HIP_a), ...]

每条星对同时存 (a,b) 和 (b,a) 两个方向，使匹配阶段能直接确定
“照片星 u -> 星表星 a”的对应关系（三角形三边闭合时方向必须一致）。

相比原项目的“三角形哈希库”（C(n,3) 个条目）：
    - 建库复杂度从 O(n³) 降到 O(n²)；
    - 匹配时每条边只需查 (2r+1) 个一维桶，不再遍历 (2r+1)³ 个三维邻桶。
三角形的“三边闭合”约束在匹配阶段完成，而不是建库阶段。
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

from .catalog import StarCatalog
from .geometry import pairwise_angular_distances_deg


def build_pair_db(catalog: StarCatalog, bin_width_deg: float) -> dict:
    """构建星对哈希库。

    参数:
        catalog: 标准星表
        bin_width_deg: 角距离分桶宽度（度）

    返回:
        {桶号: [(HIP_a, HIP_b), ...]}，含双向条目
    """
    distances = pairwise_angular_distances_deg(catalog.unit_vectors)  # (N, N)
    db = defaultdict(list)
    n = catalog.size
    for i in range(n):
        for j in range(i + 1, n):
            bucket = int(round(distances[i, j] / bin_width_deg))
            hi, hj = int(catalog.hip[i]), int(catalog.hip[j])
            db[bucket].append((hi, hj))
            db[bucket].append((hj, hi))
    return dict(db)


def save_pair_db(db: dict, path) -> None:
    """把星对哈希库保存到磁盘。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(db, f)


def load_pair_db(path) -> dict:
    """从磁盘加载星对哈希库。"""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_or_build_pair_db(catalog: StarCatalog, path, bin_width_deg: float) -> dict:
    """缓存优先：已存在则加载，否则构建并保存。

    注意：缓存与分桶宽度强相关；修改 bin_width_deg 后请删除
    cache/pair_hash_db_*.pkl（或直接换一个文件名）。
    """
    path = Path(path)
    if path.exists():
        print(f"星对哈希库已存在，直接加载：{path}")
        return load_pair_db(path)

    n_pairs = catalog.size * (catalog.size - 1) // 2
    print(f"构建星对哈希库（{catalog.size} 颗星，C(n,2)={n_pairs} 条星对）...")
    db = build_pair_db(catalog, bin_width_deg)
    save_pair_db(db, path)
    print(f"哈希库构建完成，共 {len(db)} 个桶，已保存：{path}")
    return db
