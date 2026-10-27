"""根据 detect.py 输出的 detections.csv 生成 Excel 缺陷统计报表。

报表结构：
    1. 缺陷明细   —— 每个缺陷框一行（图片/类别/置信度/像素坐标/框尺寸）
    2. 图片判定   —— 每张图一行：各类缺陷数量与 合格/不合格 判定
    3. 汇总统计   —— 总体缺陷率、各类检出数量等

用法：
    python -m src.report --detections <detections.csv> --out <报表.xlsx> [--ok-max-defects 0]
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, REPORT_OUT_DIR  # noqa: E402


def read_detections(csv_path: Path) -> list[dict]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Excel 缺陷统计报表")
    parser.add_argument("--detections", required=True, help="detect.py 输出的 detections.csv")
    parser.add_argument("--out", default=str(REPORT_OUT_DIR / "缺陷统计报表.xlsx"))
    parser.add_argument("--ok-max-defects", type=int, default=0,
                        help="单张图允许的最大缺陷数，超过判为不合格(默认0)")
    args = parser.parse_args()

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font

    rows = read_detections(Path(args.detections))
    ok_max = max(0, args.ok_max_defects)

    # 按图片聚合
    per_img: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_img[r["image"]].append(r)

    wb = Workbook()

    # Sheet1 缺陷明细
    ws1 = wb.active
    ws1.title = "缺陷明细"
    headers1 = ["图片", "类别", "置信度", "xmin(px)", "ymin(px)", "xmax(px)", "ymax(px)", "框宽(px)", "框高(px)"]
    ws1.append(headers1)
    for r in rows:
        ws1.append([
            r["image"], r["class"], r["conf"],
            r["xmin"], r["ymin"], r["xmax"], r["ymax"],
            int(r["xmax"]) - int(r["xmin"]), int(r["ymax"]) - int(r["ymin"]),
        ])

    # Sheet2 图片判定
    ws2 = wb.create_sheet("图片判定")
    headers2 = (["图片", "缺陷总数", "判定"] +
                [f"{n}数量" for n in CLASS_NAMES] + [f"{n}最大置信度" for n in CLASS_NAMES])
    ws2.append(headers2)
    ng_count = 0
    for img, dets in sorted(per_img.items()):
        cls_counts = defaultdict(int)
        cls_conf = defaultdict(float)
        for d in dets:
            cls_counts[d["class"]] += 1
            cls_conf[d["class"]] = max(cls_conf[d["class"]], float(d["conf"]))
        total = len(dets)
        verdict = "合格" if total <= ok_max else "不合格"
        if verdict == "不合格":
            ng_count += 1
        row = [img, total, verdict]
        for n in CLASS_NAMES:
            row.append(cls_counts.get(n, 0))
        for n in CLASS_NAMES:
            row.append(round(cls_conf.get(n, 0.0), 4))
        ws2.append(row)

    # Sheet3 汇总统计
    ws3 = wb.create_sheet("汇总统计")
    total_img = len(per_img)
    n_defect_img = len([v for v in per_img.values() if v])
    cls_total = defaultdict(int)
    for r in rows:
        cls_total[r["class"]] += 1
    stats = [
        ("检测图片总数", total_img),
        ("含缺陷图片数", n_defect_img),
        ("缺陷图片占比", f"{n_defect_img / total_img:.1%}" if total_img else "-"),
        ("不合格图片数", ng_count),
        ("不合格率", f"{ng_count / total_img:.1%}" if total_img else "-"),
        ("缺陷框总数", len(rows)),
        ("平均每图缺陷数", f"{len(rows) / total_img:.2f}" if total_img else "-"),
    ]
    for n in CLASS_NAMES:
        stats.append((f"{n} 检出总数", cls_total.get(n, 0)))
    for k, v in stats:
        ws3.append([k, v])

    # 轻量样式：表头加粗
    for ws in (ws1, ws2):
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="center")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out))
    print(f"[完成] 报表已生成: {out}")


if __name__ == "__main__":
    main()
