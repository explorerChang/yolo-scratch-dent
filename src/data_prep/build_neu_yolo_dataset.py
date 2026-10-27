"""从原始 NEU-DET（VOC XML 标注）构建本项目 YOLO 数据集。

做什么：
1. 只保留与项目相关的两类缺陷并重映射：
       scratches        (划痕) -> scratch (id=0)
       pitted_surface   (坑/点蚀) -> dent   (id=1)
   其余 4 类（crazing/inclusion/patches/rolled-in_scale）直接排除。
2. VOC XML -> YOLO txt（归一化 cx,cy,w,h），跳过 difficult=1 的难例框。
3. 按类别分层随机切分 train/val/test（默认 70/20/10，固定随机种子）。
4. 生成 Ultralytics 训练用的 dataset.yaml 与文件复制。

用法：
    python -m src.data_prep.build_neu_yolo_dataset [--raw-dir ...] [--out-dir ...]
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # 保证可 import src.*

from src.config import (  # noqa: E402
    DATASET_DIR,
    RAW_NEU_DIR,
    SPLIT_SEED,
    TEST_RATIO,
    VAL_RATIO,
)

# 原始类别名 -> 本项目类别 id（注意 pitted_surface 类名本身含下划线，勿用 stem 切分）
CLASS_MAP = {"scratches": 0, "pitted_surface": 1}
CLASS_BY_ID = {0: "scratches", 1: "pitted_surface"}


def parse_voc_xml(xml_path: Path):
    """解析一个 VOC XML，返回 (root, 图像宽, 图像高, 有效框列表)。

    框元素: dict(cls_id, cx, cy, w, h)，坐标已按宽高归一化并裁剪到 [0,1]。
    仅收集 difficult != 1 且类别在 CLASS_MAP 中的框。
    """
    root = ET.parse(str(xml_path)).getroot()
    size = root.find("size")
    w_img, h_img = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        diff = obj.find("difficult")
        if name not in CLASS_MAP or (diff is not None and int(diff.text) == 1):
            continue
        b = obj.find("bndbox")
        x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
        x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        # 钳制防止越界，过滤退化框
        x1 = max(0.0, min(x1, w_img - 1))
        y1 = max(0.0, min(y1, h_img - 1))
        x2 = max(0.0, min(x2, w_img))
        y2 = max(0.0, min(y2, h_img))
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            continue
        boxes.append({
            "cls_id": CLASS_MAP[name],
            "cx": (x1 + x2) / 2 / w_img,
            "cy": (y1 + y2) / 2 / h_img,
            "w": w / w_img,
            "h": h / h_img,
        })
    return root, w_img, h_img, boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 scratch/dent 两类 YOLO 数据集")
    parser.add_argument("--raw-dir", default=str(RAW_NEU_DIR), help="NEU-DET 原始数据目录（含 IMAGES/ANNOTATIONS）")
    parser.add_argument("--out-dir", default=str(DATASET_DIR), help="输出 YOLO 数据集目录")
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO, help="验证集占比")
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO, help="测试集占比")
    parser.add_argument("--seed", type=int, default=SPLIT_SEED, help="随机种子")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    ann_dir = raw_dir / "ANNOTATIONS"
    img_dir = raw_dir / "IMAGES"
    out_dir = Path(args.out_dir)
    if not ann_dir.is_dir() or not img_dir.is_dir():
        sys.exit(f"[错误] 原始数据目录结构不正确: {raw_dir}（需要含 IMAGES/ 与 ANNOTATIONS/）")

    # 1) 收集候选样本：stem -> (原始类别, 有效框列表)
    samples = []  # (stem, src_cls, img_src, boxes)
    for xml_path in sorted(ann_dir.glob("*.xml")):
        root, w_img, h_img, boxes = parse_voc_xml(xml_path)
        if not boxes:
            continue
        stem = xml_path.stem
        # NEU-DET 单图只含一种缺陷：用首个有效框推断原始类别目录
        src_cls = CLASS_BY_ID[boxes[0]["cls_id"]]
        img_src = img_dir / src_cls / f"{stem}.jpg"
        if not img_src.is_file():
            print(f"[警告] 图片缺失，跳过: {img_src}")
            continue
        samples.append((stem, src_cls, img_src, boxes))

    # 2) 按原始类别分层随机切分 train/val/test
    rng = random.Random(args.seed)
    per_cls = {c: [] for c in CLASS_MAP}
    for s in samples:
        per_cls[s[1]].append(s)
    splits = {"train": [], "val": [], "test": []}
    for c, items in per_cls.items():
        rng.shuffle(items)
        n = len(items)
        n_val = max(1, int(round(n * args.val_ratio)))
        n_test = max(1, int(round(n * args.test_ratio)))
        # 保证 train 至少留 1 张：优先削减 test，再削减 val
        overflow = n_val + n_test - (n - 1)
        if overflow > 0:
            cut = min(overflow, n_test - 1)
            n_test -= cut
            overflow -= cut
        if overflow > 0:
            n_val = max(1, n_val - overflow)
        splits["test"].extend(items[:n_test])
        splits["val"].extend(items[n_test:n_test + n_val])
        splits["train"].extend(items[n_test + n_val:])

    # 3) 复制图片并写出标签（先清理旧切分残留，避免混入上一次的文件）
    for sub in ("images", "labels"):
        old = out_dir / sub
        if old.exists():
            shutil.rmtree(old)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    stats = {"train": {0: 0, 1: 0}, "val": {0: 0, 1: 0}, "test": {0: 0, 1: 0}}
    for split, items in splits.items():
        for stem, src_cls, img_src, boxes in items:
            shutil.copy2(img_src, out_dir / "images" / split / f"{stem}.jpg")
            label_path = out_dir / "labels" / split / f"{stem}.txt"
            lines = [f"{b['cls_id']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in boxes]
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            for b in boxes:
                stats[split][b["cls_id"]] += 1

    # 4) dataset.yaml（相对路径基于该文件所在目录解析）
    yaml_text = (
        "# 由 src/data_prep/build_neu_yolo_dataset.py 自动生成\n"
        f"path: {out_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: scratch\n"
        "  1: dent\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml_text, encoding="utf-8")

    # 5) 汇总输出
    print(f"原始目录 : {raw_dir}")
    print(f"输出目录 : {out_dir}")
    for split in ("train", "val", "test"):
        n_img = len(splits[split])
        n_scratch = stats[split][0]
        n_dent = stats[split][1]
        print(f"{split:5s}: 图片 {n_img:4d} 张 | scratch 框 {n_scratch:4d} | dent 框 {n_dent:4d}")


if __name__ == "__main__":
    main()
