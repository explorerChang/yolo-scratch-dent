"""分别在 train / val / test 上评估模型，输出各自的准确率。

准确率指标：Precision、Recall、mAP50、mAP50-95（均为检测框 Box 指标）。
结果同时打印到控制台并写入 CSV。

用法：
    python -m src.evaluate                                           # 默认评估 runs/train/neu2cls/weights/best.pt
    python -m src.evaluate --weights <best.pt> --splits train,val,test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATASET_YAML, DEFAULT_IMGSZ, RUNS_DIR  # noqa: E402

DEFAULT_WEIGHTS = RUNS_DIR / "train" / "neu2cls" / "weights" / "best.pt"


def parse_args():
    p = argparse.ArgumentParser(description="按 split 评估模型准确率")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="权重 *.pt")
    p.add_argument("--data", default=str(DATASET_YAML), help="数据集 yaml")
    p.add_argument("--splits", default="train,val,test", help="要评估的 split，逗号分隔")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="auto", help="如 0, 1 或 cpu")
    p.add_argument("--out", default=str(RUNS_DIR / "eval" / "result.csv"), help="评估结果 CSV 输出路径")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        sys.exit(f"[错误] 权重不存在: {weights}（先训练，或 --weights 指定其它 *.pt）")

    from ultralytics import YOLO

    device = args.device
    if device == "auto":
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    model = YOLO(str(weights))

    eval_dir = RUNS_DIR / "eval"  # 与其他产物统一放在项目根 runs 下
    rows = []
    for split in splits:
        m = model.val(
            data=args.data,
            split=split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=str(eval_dir),
            name=split,
            exist_ok=True,
            verbose=False,
        )
        d = m.results_dict
        rows.append((
            split,
            d["metrics/precision(B)"],
            d["metrics/recall(B)"],
            d["metrics/mAP50(B)"],
            d["metrics/mAP50-95(B)"],
        ))

    print(f"\n权重: {weights}\n")
    print(f"{'数据集':<8}{'Precision':>12}{'Recall':>12}{'mAP50':>12}{'mAP50-95':>12}")
    for split, p, r, m50, m5095 in rows:
        print(f"{split:<8}{p:>12.4f}{r:>12.4f}{m50:>12.4f}{m5095:>12.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "precision", "recall", "mAP50", "mAP50-95"])
        for split, p, r, m50, m5095 in rows:
            writer.writerow([split, f"{p:.4f}", f"{r:.4f}", f"{m50:.4f}", f"{m5095:.4f}"])
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
