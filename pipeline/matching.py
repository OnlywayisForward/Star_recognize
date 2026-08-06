# -*- coding: utf-8 -*-
"""三角形匹配：投票 + 全局验证 + 扩展匹配。

原理：
    三角形由三条边的角距离唯一确定（忽略镜像）。流程如下——

    1. 枚举照片中最亮前 K 颗星组成的所有三角形（C(K,3) 个）；
    2. 对每条边，用角距离在星对哈希库中查桶（含 ±neighbor_bins 邻域），
       得到若干候选“有序星对”(a,b)，含义是 照片星u -> 星表星a、照片星v -> 星表星b；
    3. 若三条边的候选星对能首尾相接（(a,b)、(b,c)、(c,a)），
       则 (a,b,c) 是一个与照片三角形一致的标准星三元组；
    4. 每得到一个一致三元组，就为三颗照片星各投一票；
    5. 投票数超过阈值的映射作为初步匹配；
    6. 用 Kabsch 求最优旋转验证全局一致性，剔除残差过大的误匹配；
    7. 用旋转矩阵把其余照片星投影到天球，在更全的星表中找最近邻，完成扩展匹配。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

from .catalog import StarCatalog
from .geometry import kabsch_rotation, pairwise_angular_distances_deg


@dataclass
class MatchResult:
    """匹配结果。"""

    photo_to_hip: dict = field(default_factory=dict)      # {照片星下标: HIP 编号}
    rotation: np.ndarray = None                           # 照片系 -> 星表系的旋转矩阵
    residual_deg: float = None                            # 验证后的平均残差（度）
    votes_per_star: dict = field(default_factory=dict)    # 每颗照片星的投票分布（调试用）
    stats: dict = field(default_factory=dict)             # 各阶段数量统计


def match_stars(photo_vectors, pair_db, bin_width_deg, neighbor_bins=5,
                photo_top_n=40, min_votes=2) -> MatchResult:
    """三角形投票匹配（全局验证由 verify_and_extend 完成）。

    参数:
        photo_vectors: 照片星单位向量，(N,3)
        pair_db: 星对哈希库
        bin_width_deg: 分桶宽度（度）
        neighbor_bins: 查桶时向左右各扩展几个桶
        photo_top_n: 只枚举最亮前 K 颗星组成的三角形
        min_votes: 投票阈值

    返回:
        MatchResult
    """
    photo_vectors = np.asarray(photo_vectors, dtype=float)
    n_photo = len(photo_vectors)
    top_n = min(photo_top_n, n_photo)

    result = MatchResult(stats={"photo_stars": n_photo, "top_n_used": top_n})
    if top_n < 3:
        result.stats.update(total_triangles=0, matched_triangles=0)
        return result

    # 预计算最亮前 K 颗星的两两角距离矩阵，三角形边角直接查表，避免重复反余弦
    dist = pairwise_angular_distances_deg(photo_vectors[:top_n])

    votes = defaultdict(Counter)
    matched_triangles = 0
    total_triangles = top_n * (top_n - 1) * (top_n - 2) // 6

    for i, j, k in combinations(range(top_n), 3):
        triples = _consistent_triples(
            dist[i, j], dist[j, k], dist[k, i],
            pair_db, bin_width_deg, neighbor_bins,
        )
        if not triples:
            continue
        matched_triangles += 1
        for a, b, c in triples:
            votes[i][a] += 1
            votes[j][b] += 1
            votes[k][c] += 1

    # 投票阈值过滤：每颗照片星取票数最多的 HIP
    photo_to_hip = {}
    for photo_idx, counter in votes.items():
        hip, count = counter.most_common(1)[0]
        result.votes_per_star[photo_idx] = dict(counter)
        if count >= min_votes:
            photo_to_hip[photo_idx] = hip

    result.photo_to_hip = photo_to_hip
    result.stats.update(
        total_triangles=total_triangles,
        matched_triangles=matched_triangles,
        voted_matches=len(photo_to_hip),
    )
    return result


def verify_and_extend(result: MatchResult, photo_vectors, catalog: StarCatalog,
                      extension_catalog: StarCatalog = None,
                      verify_residual_deg=1.0, extend_residual_deg=0.5,
                      max_iter=3) -> MatchResult:
    """Kabsch 全局验证 + 扩展匹配。

    步骤：
        1. 用当前映射求最优旋转，做鲁棒性迭代：
           先按“中位数 + 3σ”粗阈值剔除明显离群点（个别错误投票会把
           最小二乘拟合带偏，用固定阈值会误伤好点），再重拟合、收敛；
        2. 若提供了更全的星表，把全部照片星旋转投影到天球，
           在星表中找最近邻，角距小于 extend_residual_deg 则补充为匹配。
    """
    photo_vectors = np.asarray(photo_vectors, dtype=float)
    mapping = dict(result.photo_to_hip)

    # ---------- 1. 迭代验证 ----------
    rotation, residual = None, None
    for _ in range(max_iter):
        if len(mapping) < 3:
            rotation, residual = None, None
            break
        indices = list(mapping.keys())
        src = photo_vectors[indices]
        dst = _vectors_for_hips(catalog, [mapping[i] for i in indices])
        rotation, residual = kabsch_rotation(src, dst)

        # 鲁棒阈值：中位数 + 3σ（σ 用 1.4826·MAD 估计），但不低于硬性阈值
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        cutoff = max(verify_residual_deg, median + 3.0 * 1.4826 * mad)

        filtered = {
            idx: hip
            for idx, hip, res in zip(indices, mapping.values(), residual)
            if res <= cutoff
        }
        if len(filtered) == len(mapping):      # 本轮没有剔除，已收敛
            mapping = filtered
            break
        mapping = filtered

    mapping = {int(idx): int(hip) for idx, hip in mapping.items()}
    result.photo_to_hip = mapping
    result.rotation = rotation
    result.residual_deg = float(np.mean(residual)) if residual is not None else None
    result.stats["verified_matches"] = len(mapping)

    # ---------- 2. 扩展匹配（可选） ----------
    if extension_catalog is not None and rotation is not None:
        before = len(mapping)
        mapping = _extend_matches(
            photo_vectors, mapping, rotation,
            extension_catalog, extend_residual_deg,
        )
        result.photo_to_hip = mapping
        result.stats["extended_matches"] = len(mapping) - before

    return result


def _consistent_triples(d12, d23, d31, pair_db, bin_width_deg, neighbor_bins):
    """给定照片三角形三条边的角距离，返回所有一致的标准星三元组。

    照片三角形 (u, v, w) 的三条边约定为：d12=u-v，d23=v-w，d31=w-u。
    标准三元组 (a, b, c) 与它一致，当且仅当：
        (a,b) ∈ d12 的候选、(b,c) ∈ d23 的候选、(c,a) ∈ d31 的候选。
    """
    c12 = _lookup_pairs(d12, pair_db, bin_width_deg, neighbor_bins)
    c23 = _lookup_pairs(d23, pair_db, bin_width_deg, neighbor_bins)
    c31 = _lookup_pairs(d31, pair_db, bin_width_deg, neighbor_bins)
    if not (c12 and c23 and c31):
        return []

    # 建两个小索引，避免三重循环
    out23 = defaultdict(set)   # b -> {c | (b,c) ∈ c23}
    in31 = defaultdict(set)    # a -> {c | (c,a) ∈ c31}
    for b, c in c23:
        out23[b].add(c)
    for c, a in c31:
        in31[a].add(c)

    triples = []
    for a, b in c12:
        for c in out23.get(b, ()):
            if c in in31.get(a, ()):
                triples.append((a, b, c))
    return triples


def _lookup_pairs(edge_deg, pair_db, bin_width_deg, neighbor_bins):
    """按角距离查桶（含左右邻域），返回该边对应的候选有序星对列表。"""
    center = int(round(edge_deg / bin_width_deg))
    result = []
    for offset in range(-neighbor_bins, neighbor_bins + 1):
        result.extend(pair_db.get(center + offset, ()))
    return result


def _vectors_for_hips(catalog: StarCatalog, hips) -> np.ndarray:
    """按 HIP 列表取星表单位向量（编号必须都存在）。"""
    rows = []
    for h in hips:
        idx = catalog.index_of_hip(h)
        if idx < 0:
            raise KeyError(f"HIP {h} 不在星表中")
        rows.append(catalog.unit_vectors[idx])
    return np.asarray(rows)


def _extend_matches(photo_vectors, mapping, rotation, extension_catalog,
                    threshold_deg) -> dict:
    """把照片星旋转到天球坐标系，在星表中找最近邻补充匹配。

    保证同一颗星表星最多被一颗照片星认领；按角距从小到大处理，
    让更可靠的匹配优先占用星表星。
    """
    rotated = photo_vectors @ rotation.T

    try:
        from scipy.spatial import cKDTree

        # 弦距离阈值与角距阈值等价：chord = 2·sin(θ/2)
        chord_thresh = 2.0 * np.sin(np.radians(threshold_deg) / 2.0)
        dist, idx = cKDTree(extension_catalog.unit_vectors).query(rotated)
        keep = dist <= chord_thresh
    except ImportError:
        # 无 scipy 时退化为暴力最近邻（约 5000 颗星，仍然很快）
        dots = rotated @ extension_catalog.unit_vectors.T
        dist = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
        idx = np.argmax(dots, axis=1)
        keep = dist <= threshold_deg

    extended = dict(mapping)
    used_hips = set(mapping.values())
    for photo_idx in np.argsort(dist, kind="stable"):
        if not keep[photo_idx]:
            break                       # 之后只会更远
        if photo_idx in extended:
            continue
        hip = int(extension_catalog.hip[idx[photo_idx]])
        if hip in used_hips:
            continue                    # 该星表星已被别的照片星认领
        extended[photo_idx] = hip
        used_hips.add(hip)
    return extended
