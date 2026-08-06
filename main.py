# -*- coding: utf-8 -*-
"""星图识别主流程（重构版）。

用法:
    python main.py [图片路径]

流程:
    照片 -> 星点检测（像素坐标）-> 针孔模型转单位方向向量
        -> 与 3 等星表构建的星对哈希库做三角形匹配 + 投票
        -> Kabsch 旋转对齐验证
        -> 用 6 等星表扩展识别剩余星点
        -> 在原图上标注 HIP 编号并输出
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import config
from pipeline import camera, catalog, database, image, matching, visualization


def _resolve_catalog_csv() -> Path:
    """优先使用原项目 data 目录中已有的 CSV（只读）；没有则自己提取一份到 cache。"""
    if config.CATALOG_CSV.exists():
        return config.CATALOG_CSV
    if config.CATALOG_CSV_CACHE.exists():
        return config.CATALOG_CSV_CACHE
    if not config.CATALOG_TSV.exists():
        raise FileNotFoundError(f"找不到星表文件：{config.CATALOG_CSV} 或 {config.CATALOG_TSV}")
    return catalog.extract_cdata_to_csv(config.CATALOG_TSV, config.CATALOG_CSV_CACHE)


def main(image_path=None) -> None:
    t_start = time.perf_counter()
    image_path = Path(image_path) if image_path else config.SAMPLE_IMAGE

    # ---------- 1. 星点检测 ----------
    print("\n[1/6] 星点检测")
    detection = image.detect_star_points(
        image_path,
        threshold=config.DETECTION_THRESHOLD,
        min_distance=config.DETECTION_MIN_DISTANCE,
        save_debug=config.DETECTION_SAVE_DEBUG,
        debug_dir=config.DEBUG_DIR,
    )
    star_points = detection.points
    print(f"      检测到 {len(star_points)} 颗星点（按亮度降序）")
    if len(star_points) < 3:
        sys.exit(f"星点太少（<3），无法匹配。可调低 DETECTION_THRESHOLD（当前 {config.DETECTION_THRESHOLD}）。")

    # ---------- 2. 像素 -> 观测方向单位向量 ----------
    print("[2/6] 像素坐标 -> 单位方向向量")
    f_px = camera.focal_length_px(config.FOCAL_MM, config.PIXEL_SIZE_MM)
    photo_vectors = camera.pixels_to_unit_vectors(
        star_points, detection.width, detection.height, f_px,
    )
    print(f"      图像 {detection.width}x{detection.height}，焦距 {f_px:.1f}px")

    # ---------- 3. 加载星表 + 构建/加载星对哈希库 ----------
    print("[3/6] 星表与星对哈希库")
    csv_path = _resolve_catalog_csv()
    star_catalog = catalog.load_catalog(csv_path, max_stars=config.CATALOG_TOP_N)
    print(f"      星表：{star_catalog.size} 颗星（≤3 等）")
    pair_db = database.load_or_build_pair_db(
        star_catalog, config.PAIR_DB_PATH, config.BIN_WIDTH_DEG,
    )

    # ---------- 4. 三角形匹配 + 投票 ----------
    print("[4/6] 三角形匹配 + 投票")
    result = matching.match_stars(
        photo_vectors, pair_db,
        bin_width_deg=config.BIN_WIDTH_DEG,
        neighbor_bins=config.NEIGHBOR_BINS,
        photo_top_n=config.PHOTO_TOP_N,
        min_votes=config.MIN_VOTES,
    )
    print(f"      枚举三角形 {result.stats['total_triangles']} 个，"
          f"命中 {result.stats['matched_triangles']} 个")
    print(f"      投票后初步匹配 {len(result.photo_to_hip)} 颗")

    # ---------- 5. 全局验证 + 扩展匹配 ----------
    print("[5/6] Kabsch 全局验证")
    extension_catalog = None
    if config.ENABLE_EXTENSION and config.EXTENSION_CSV.exists():
        extension_catalog = catalog.load_catalog(config.EXTENSION_CSV)
        print(f"      扩展星表：{extension_catalog.size} 颗星（≤6 等）")
    else:
        print("      未启用扩展匹配（未找到 6 等星表）")

    result = matching.verify_and_extend(
        result, photo_vectors, star_catalog, extension_catalog,
        verify_residual_deg=config.VERIFY_RESIDUAL_DEG,
        extend_residual_deg=config.EXTEND_RESIDUAL_DEG,
        max_iter=config.VERIFY_MAX_ITER,
    )
    if result.residual_deg is None:
        print("      匹配点不足，无法做旋转验证")
    else:
        print(f"      验证后共匹配 {len(result.photo_to_hip)} 颗，"
              f"平均残差 {result.residual_deg:.3f}°")

    # ---------- 6. 标注输出 ----------
    print("[6/6] 标注输出")
    visualization.draw_star_points(
        image_path, star_points, config.OUTPUT_DIR / "star_points_marked.jpg",
    )
    visualization.draw_hip_labels(
        image_path, star_points, result.photo_to_hip,
        config.OUTPUT_DIR / "matched_result.jpg",
    )

    elapsed = time.perf_counter() - t_start
    print(f"\n完成，耗时 {elapsed:.1f}s。输出目录：{config.OUTPUT_DIR}")
    if not result.photo_to_hip:
        sys.exit("警告：没有匹配到任何恒星。请检查相机参数（FOCAL_MM / PIXEL_SIZE_MM）。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="星图识别（重构版）")
    parser.add_argument("image", nargs="?", default=None,
                        help="照片路径，默认 data/Antares.jpg")
    args = parser.parse_args()
    main(Path(args.image) if args.image else config.SAMPLE_IMAGE)
