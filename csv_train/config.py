"""
全局配置：路径、常量、matplotlib 中文字体
"""

import os
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── 中文字体（Windows: SimHei / 黑体） ────────────────────────────────────────
try:
    matplotlib.rcParams["font.family"] = "SimHei"
    matplotlib.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

# ── 路径配置 ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data_csv", "processed")
FIG_DIR    = os.path.join(SCRIPT_DIR, "output")
MODEL_DIR  = os.path.join(SCRIPT_DIR, "models")
META_PATH  = os.path.join(DATA_DIR, "dataset_metadata.json")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── 标签 & 特征 ───────────────────────────────────────────────────────────────
CLASS_NAMES = ["无病(0)", "轻度(1)", "中度(2)", "重度(3)"]
LABEL_COL   = "disease_level"
DROP_COLS   = ["station", "date", "year", "disease_risk_score",
               "disease_level", "disease_level_name"]

# ── 绘图调色板 ────────────────────────────────────────────────────────────────
PALETTE = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4"]
