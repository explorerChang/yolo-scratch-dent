"""训练 YOLO 检测模型（Ultralytics）。

用法示例：
    python -m src.train --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16
    python -m src.train --model runs/train/neu2cls/weights/best.pt  # 断点续训/微调
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    DATASET_YAML,
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_MODEL,
    RUNS_DIR,
)


def parse_args():
    p = argparse.ArgumentParser(description="训练 YOLO 表面缺陷检测模型")
    p.add_argument("--data", default=str(DATASET_YAML), help="数据集 yaml")
    p.add_argument("--model", default=DEFAULT_MODEL, help="预训练权重或上次训练权重")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    p.add_argument("--device", default="auto", help="如 0, 1 或 cpu")
    p.add_argument("--name", default="neu2cls", help="本次运行名（决定输出目录）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    # CPU-only torch 不支持 device=auto：有 GPU 用 0，无 GPU 回退 cpu
    device = args.device
    if device == "auto":
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(RUNS_DIR / "train"),
        name=args.name,
        patience=20,
        seed=0,
        plots=True,
    )
    best = getattr(model.trainer, "best", RUNS_DIR / "train" / args.name / "weights" / "best.pt")
    print(f"[完成] 最优权重: {best}")


if __name__ == "__main__":
    main()
