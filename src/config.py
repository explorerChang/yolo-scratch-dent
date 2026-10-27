"""项目级配置：路径、类别定义与默认训练参数。

约定统一从这里取路径/类别，避免各脚本硬编码漂移。
"""
from __future__ import annotations

import os
from pathlib import Path

# 项目根目录（src/config.py 的上上级）
ROOT = Path(__file__).resolve().parents[1]

# 本项目固定的两类缺陷：scratch=划痕, dent=凹坑
CLASS_NAMES = ["scratch", "dent"]
CLASS_IDS = {name: i for i, name in enumerate(CLASS_NAMES)}

# 原始数据集：默认取桌面上的 NEU-DET 镜像，可用环境变量 RAW_NEU_DIR 覆盖
_default_raw = Path(os.environ.get("USERPROFILE", "")) / "Desktop" / "NEU-DET"
RAW_NEU_DIR = Path(os.environ.get("RAW_NEU_DIR", str(_default_raw)))

# 生成后的 YOLO 格式数据集（scratch/dent 两类）
DATASET_DIR = ROOT / "data" / "neu-det-2cls"
DATASET_YAML = DATASET_DIR / "dataset.yaml"

WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"
DETECT_OUT_DIR = RUNS_DIR / "detect"
REPORT_OUT_DIR = RUNS_DIR / "report"

# 数据切分
SPLIT_SEED = 42
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 训练默认参数（本机有 NVIDIA GPU，启动可用命令行覆盖）
DEFAULT_MODEL = "yolov8n.pt"  # 预训练起点
DEFAULT_IMGSZ = 640
DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16

# 推理默认参数
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
