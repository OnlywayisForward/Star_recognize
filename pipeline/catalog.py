# -*- coding: utf-8 -*-
"""星表加载与解析。

原始数据是 VizieR 导出的 VOTable(TSV)，表格主体在 <![CDATA[...]]> 中，
是分号分隔文本，前三行分别是表头、单位、分隔线。本模块负责：
    1) 从 VOTable 中提取 CDATA 存为 CSV；
    2) 把 CSV 读成结构化星表（HIP、赤经、赤纬、单位向量）。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry

# VizieR 表格中我们关心的列
_REQUIRED_COLS = ["HIP", "_RA.icrs", "_DE.icrs"]
_MAG_COL = "Vmag"


@dataclass
class StarCatalog:
    """一颗星的字段：HIP 编号、赤经（度）、赤纬（度）、单位方向向量。"""

    hip: np.ndarray           # (N,)
    ra_deg: np.ndarray        # (N,)
    dec_deg: np.ndarray       # (N,)
    unit_vectors: np.ndarray  # (N,3)

    @property
    def size(self) -> int:
        return len(self.hip)

    def index_of_hip(self, hip_id: int) -> int:
        """按 HIP 编号查其在数组中的下标；不存在时返回 -1。"""
        matches = np.flatnonzero(self.hip == hip_id)
        return int(matches[0]) if len(matches) else -1


def extract_cdata_to_csv(input_path, output_path) -> Path:
    """从 VizieR VOTable 的 <![CDATA[...]]> 中取出表格文本，另存为 CSV。

    参数:
        input_path: VOTable 文件路径（.tsv）
        output_path: 输出 CSV 路径

    返回:
        输出文件路径
    """
    input_path, output_path = Path(input_path), Path(output_path)
    text = input_path.read_text(encoding="utf-8")

    start = text.find("<![CDATA[")
    end = text.find("]]>")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"在 {input_path} 中找不到 CDATA 数据块")

    cdata = text[start + len("<![CDATA[") : end].strip()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cdata, encoding="utf-8")
    print(f"已从 {input_path} 提取星表数据到 {output_path}")
    return output_path


def load_catalog(csv_path, max_stars=None) -> StarCatalog:
    """读取 VizieR 星表 CSV，返回 StarCatalog。

    参数:
        csv_path: 星表 CSV 路径（分号分隔，前三行为表头/单位/分隔线）
        max_stars: 若指定，则按视星等 Vmag 升序取最亮的前 N 颗

    返回:
        StarCatalog
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path, sep=";", skiprows=[1, 2])

    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"星表缺少必要列 {missing}，请检查文件格式：{csv_path}")

    df = df[_REQUIRED_COLS + ([_MAG_COL] if _MAG_COL in df.columns else [])]
    df[_REQUIRED_COLS] = df[_REQUIRED_COLS].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=_REQUIRED_COLS, inplace=True)
    df["HIP"] = df["HIP"].astype(int)

    if max_stars is not None and _MAG_COL in df.columns:
        df = df.sort_values(_MAG_COL, na_position="last").head(max_stars)

    hip = df["HIP"].to_numpy(dtype=np.int64)
    ra = df["_RA.icrs"].to_numpy(dtype=float)
    dec = df["_DE.icrs"].to_numpy(dtype=float)
    return StarCatalog(
        hip=hip,
        ra_deg=ra,
        dec_deg=dec,
        unit_vectors=geometry.radec_to_unit_vectors(ra, dec),
    )
