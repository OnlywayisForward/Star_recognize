# -*- coding: utf-8 -*-
"""全局配置：路径与算法参数集中管理。

所有路径都基于本文件所在目录（Reconstruction/）推导，
因此无论从哪里运行 main.py，都能正确定位到数据与输出目录。
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# 目录
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent               # Reconstruction/
DATA_DIR = PROJECT_ROOT.parent / "data"                      # 与原项目共用 E:\StarMapper\data（只读）
OUTPUT_DIR = PROJECT_ROOT / "output"                         # 重构版输出目录（不碰原项目 output）
DEBUG_DIR = OUTPUT_DIR / "debug"                             # 调试中间图（可选）
CACHE_DIR = PROJECT_ROOT / "cache"                           # 哈希库等缓存

# ---------------------------------------------------------------------------
# 数据文件
# ---------------------------------------------------------------------------
CATALOG_TSV = DATA_DIR / "vizier_votable_maxVmags_3.tsv"               # 原始 VizieR VOTable
CATALOG_CSV = DATA_DIR / "stars_clean_vizier_votable_maxVmags_3.csv"   # 原项目已提取的 3 等星表（只读）
CATALOG_CSV_CACHE = CACHE_DIR / "stars_clean_vizier_votable_maxVmags_3.csv"  # 缺失时本项目自备一份
EXTENSION_CSV = DATA_DIR / "stars_clean_vizier_votable_maxVmags_6.csv"      # 6 等星表（扩展匹配用）

SAMPLE_IMAGE = DATA_DIR / "Antares.jpg"                      # 默认测试照片

# ---------------------------------------------------------------------------
# 星点检测
# ---------------------------------------------------------------------------
DETECTION_THRESHOLD = 252        # 亮区二值化阈值：只保留接近饱和的星点核（255 为饱和）
DETECTION_MIN_DISTANCE = 10      # 两星之间的最小像素间距，避免同一颗星被重复检测
DETECTION_SAVE_DEBUG = False     # 是否保存调试中间图（原项目会固定写 Trash/ 目录）

# ---------------------------------------------------------------------------
# 相机模型（针孔模型）。主点取图像中心、图像宽高从照片实际读取；
# 下面两个尺寸仅作为读取失败时的兜底值。
# ---------------------------------------------------------------------------
IMG_WIDTH = 4000                 # 图像宽度（像素）
IMG_HEIGHT = 1600                # 图像高度（像素）
FOCAL_MM = 26.0                  # 焦距（毫米）
PIXEL_SIZE_MM = 0.0085           # 像素物理尺寸（毫米/像素，即 8.5 µm）

# ---------------------------------------------------------------------------
# 星对哈希库与三角形匹配
# ---------------------------------------------------------------------------
BIN_WIDTH_DEG = 0.01             # 角距离分桶宽度（度）：0.01° ≈ 36″
NEIGHBOR_BINS = 5                # 查桶时向左右各扩展几个桶，吸收测量噪声（±0.05°）
CATALOG_TOP_N = None             # 建库时用星表最亮的前 N 颗；None 表示全部使用
PHOTO_TOP_N = 40                 # 匹配时使用照片中最亮的前 N 颗（星点已按亮度降序）
MIN_VOTES = 2                    # 投票阈值：同一照片星至少 N 次投给同一个 HIP 才保留

# 星对哈希库缓存文件（重构版自己的格式，与原项目的 triangle_hash_db 无关；
# 文件名带上桶宽，改了 BIN_WIDTH_DEG 会自动用新缓存）
PAIR_DB_PATH = CACHE_DIR / f"pair_hash_db_maxVmags_3_bin{BIN_WIDTH_DEG:g}.pkl"

# ---------------------------------------------------------------------------
# 全局验证与扩展匹配
# ---------------------------------------------------------------------------
ENABLE_EXTENSION = True          # 是否用更全的 6 等星表把剩余照片星也识别出来
VERIFY_RESIDUAL_DEG = 1.0        # Kabsch 验证残差阈值（度），超过则视为误匹配剔除
EXTEND_RESIDUAL_DEG = 0.5        # 扩展匹配的最近邻角距阈值（度）
VERIFY_MAX_ITER = 5              # 验证-剔除循环最多迭代次数
