"""批量推理：对图片目录/单张图片执行 YOLO 检测。

输出：
1. 带检测框（类别+置信度）的标注图 -> <out>/annotated/
2. 每个缺陷框的结构化记录  -> <out>/detections.csv（供 report.py 生成 Excel）

用法：
    python -m src.detect --weights <best.pt> --source <图片目录|单张图> --out <输出目录>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (  # noqa: E402
    CLASS_NAMES,
    DEFAULT_CONF,
    DEFAULT_IOU,
    DETECT_OUT_DIR,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 批量表面缺陷检测")
    parser.add_argument("--weights", required=True, help="训练好的权重 *.pt")
    parser.add_argument("--source", required=True, help="待检图片或目录")
    parser.add_argument("--out", default=str(DETECT_OUT_DIR), help="输出目录")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU 阈值")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    images = collect_images(Path(args.source))
    if not images:
        sys.exit(f"[错误] 没有找到可检测的图片: {args.source}")

    device = args.device
    if device == "auto":
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out)
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for img_path in images:
        res = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=device,
            verbose=False,
        )[0]
        # 保存标注图
        res.save(filename=str(ann_dir / img_path.name))
        # 记录缺陷框（像素坐标）
        if res.boxes is not None:
            names = res.names
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append({
                    "image": img_path.name,
                    "cls_id": int(box.cls[0]),
                    "class": names[int(box.cls[0])],
                    "conf": round(float(box.conf[0]), 4),
                    "xmin": int(x1), "ymin": int(y1),
                    "xmax": int(x2), "ymax": int(y2),
                })

    csv_path = out_dir / "detections.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
            "image", "cls_id", "class", "conf", "xmin", "ymin", "xmax", "ymax"])
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter
    cls_counter = Counter(r["class"] for r in rows)
    n_img_defect = len({r["image"] for r in rows})
    print(f"[完成] 检测图片 {len(images)} 张 | 含缺陷图 {n_img_defect} 张 | 缺陷框 {len(rows)} 个")
    for name in CLASS_NAMES:
        print(f"        {name}: {cls_counter.get(name, 0)} 个")
    print(f"标注图   : {ann_dir}")
    print(f"记录文件 : {csv_path}")


if __name__ == "__main__":
    main()
