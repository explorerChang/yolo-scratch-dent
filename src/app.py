"""表面缺陷检测 —— 桌面图形界面（tkinter，无需控制台操作）。

功能：
    1. 选择单张/多张图片，或整个文件夹
    2. 调节置信度阈值后检测，界面左右对照显示"原图 / 检测结果"
    3. 上一张 / 下一张 浏览
    4. 一键批量检测并保存：标注图 + detections.csv

运行：
    python -m src.app
"""
from __future__ import annotations

import csv
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CLASS_NAMES, DEFAULT_CONF, DETECT_OUT_DIR, RUNS_DIR  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_WEIGHTS = RUNS_DIR / "train" / "neu2cls" / "weights" / "best.pt"
MAX_W, MAX_H = 520, 400  # 预览区最大显示尺寸


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.model = None
        self.images: list[Path] = []
        self.idx = 0
        self.rows: list[dict] = []       # 当前图的检测框
        self._photo_orig = None          # 防止被 GC
        self._photo_pred = None

        root.title("表面缺陷检测（scratch / dent）")
        root.geometry("1180x720")

        self._build_toolbar()
        self._build_preview()
        self._build_result_area()
        self._build_statusbar()

        self.status.set(f"权重: {DEFAULT_WEIGHTS}")
        if not DEFAULT_WEIGHTS.is_file():
            self.status.set(f"[警告] 权重不存在: {DEFAULT_WEIGHTS}")

    # ---------- 界面搭建 ----------
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self.root, padding=8)
        bar.pack(fill=tk.X)

        ttk.Button(bar, text="选择图片", command=self.pick_files).pack(side=tk.LEFT)
        ttk.Button(bar, text="选择文件夹", command=self.pick_dir).pack(side=tk.LEFT, padx=6)

        ttk.Label(bar, text="置信度阈值").pack(side=tk.LEFT, padx=(16, 4))
        self.conf = tk.DoubleVar(value=DEFAULT_CONF)
        ttk.Scale(bar, from_=0.05, to=0.9, variable=self.conf,
                  orient=tk.HORIZONTAL, length=160).pack(side=tk.LEFT)
        self.conf_label = ttk.Label(bar, text=f"{DEFAULT_CONF:.2f}", width=5)
        self.conf_label.pack(side=tk.LEFT)
        self.conf.trace_add("write", lambda *_: self.conf_label.config(
            text=f"{self.conf.get():.2f}"))

        ttk.Button(bar, text="检测当前图", command=self.detect_current).pack(side=tk.LEFT, padx=(16, 6))
        ttk.Button(bar, text="批量检测并保存", command=self.detect_all).pack(side=tk.LEFT)

    def _build_preview(self) -> None:
        box = ttk.Frame(self.root, padding=(8, 0))
        box.pack(fill=tk.BOTH, expand=True)

        left = ttk.Labelframe(box, text="原图", padding=6)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_orig = tk.Canvas(left, width=MAX_W, height=MAX_H,
                                     background="#f0f0f0", highlightthickness=0)
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)

        right = ttk.Labelframe(box, text="检测结果", padding=6)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        self.canvas_pred = tk.Canvas(right, width=MAX_W, height=MAX_H,
                                     background="#f0f0f0", highlightthickness=0)
        self.canvas_pred.pack(fill=tk.BOTH, expand=True)

        nav = ttk.Frame(self.root, padding=(8, 4))
        nav.pack(fill=tk.X)
        ttk.Button(nav, text="◀ 上一张", command=lambda: self.step(-1)).pack(side=tk.LEFT)
        ttk.Button(nav, text="下一张 ▶", command=lambda: self.step(1)).pack(side=tk.LEFT, padx=6)
        self.file_label = ttk.Label(nav, text="未选择图片")
        self.file_label.pack(side=tk.LEFT, padx=12)

    def _build_result_area(self) -> None:
        area = ttk.Labelframe(self.root, text="检测结果", padding=6)
        area.pack(fill=tk.BOTH, padx=8, pady=(0, 4))
        cols = ("class", "conf", "xmin", "ymin", "xmax", "ymax")
        self.tree = ttk.Treeview(area, columns=cols, show="headings", height=6)
        for c, t, w in (("class", "类别", 90), ("conf", "置信度", 90),
                        ("xmin", "xmin", 80), ("ymin", "ymin", 80),
                        ("xmax", "xmax", 80), ("ymax", "ymax", 80)):
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor=tk.CENTER)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def _build_statusbar(self) -> None:
        self.status = tk.StringVar(value="就绪")
        ttk.Label(self.root, textvariable=self.status, relief=tk.SUNKEN,
                  anchor=tk.W, padding=4).pack(fill=tk.X, side=tk.BOTTOM)

    # ---------- 选图 ----------
    def pick_files(self) -> None:
        files = filedialog.askopenfilenames(
            title="选择图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")],
        )
        if files:
            self._set_images([Path(f) for f in files])

    def pick_dir(self) -> None:
        d = filedialog.askdirectory(title="选择图片文件夹")
        if not d:
            return
        imgs = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not imgs:
            messagebox.showinfo("提示", "该文件夹下没有找到图片")
            return
        self._set_images(imgs)

    def _set_images(self, imgs: list[Path]) -> None:
        self.images = imgs
        self.idx = 0
        self.detect_current()

    def step(self, delta: int) -> None:
        if not self.images:
            return
        self.idx = (self.idx + delta) % len(self.images)
        self.detect_current()

    # ---------- 检测 ----------
    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if not DEFAULT_WEIGHTS.is_file():
            messagebox.showerror("错误", f"权重文件不存在：\n{DEFAULT_WEIGHTS}")
            return False
        self.status.set("正在加载模型…")
        self.root.update_idletasks()
        from ultralytics import YOLO
        self.model = YOLO(str(DEFAULT_WEIGHTS))
        return True

    def _device(self):
        import torch
        return 0 if torch.cuda.is_available() else "cpu"

    def detect_current(self) -> None:
        if not self.images:
            return
        if not self._ensure_model():
            return
        img = self.images[self.idx]
        self.status.set(f"检测中 ({self.idx + 1}/{len(self.images)}): {img.name}")
        self.root.update_idletasks()

        res = self.model.predict(source=str(img), conf=float(self.conf.get()),
                                 device=self._device(), verbose=False)[0]

        self._show_image(self.canvas_orig, img, "orig")
        self._show_array(self.canvas_pred, res.plot())

        self.rows = []
        names = res.names
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                self.rows.append({
                    "image": img.name,
                    "cls_id": int(box.cls[0]),
                    "class": names[int(box.cls[0])],
                    "conf": round(float(box.conf[0]), 4),
                    "xmin": int(x1), "ymin": int(y1),
                    "xmax": int(x2), "ymax": int(y2),
                })

        self.tree.delete(*self.tree.get_children())
        for r in self.rows:
            self.tree.insert("", tk.END, values=(r["class"], r["conf"],
                             r["xmin"], r["ymin"], r["xmax"], r["ymax"]))
        self.file_label.config(text=f"{img.name}  ({self.idx + 1}/{len(self.images)})")

        cls_count = {n: sum(1 for r in self.rows if r["class"] == n) for n in CLASS_NAMES}
        detail = " | ".join(f"{n}: {c}" for n, c in cls_count.items())
        self.status.set(f"完成：{img.name} → 检出 {len(self.rows)} 个（{detail}）"
                        if self.rows else f"完成：{img.name} → 未检出缺陷")

    def detect_all(self) -> None:
        if not self.images:
            messagebox.showinfo("提示", "请先选择图片或文件夹")
            return
        if not self._ensure_model():
            return

        ann_dir = DETECT_OUT_DIR / "annotated"
        ann_dir.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict] = []

        for i, img in enumerate(self.images, 1):
            self.status.set(f"批量检测中 ({i}/{len(self.images)}): {img.name}")
            self.root.update_idletasks()
            res = self.model.predict(source=str(img), conf=float(self.conf.get()),
                                     device=self._device(), verbose=False)[0]
            res.save(filename=str(ann_dir / img.name))
            if res.boxes is not None:
                names = res.names
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    all_rows.append({
                        "image": img.name,
                        "cls_id": int(box.cls[0]),
                        "class": names[int(box.cls[0])],
                        "conf": round(float(box.conf[0]), 4),
                        "xmin": int(x1), "ymin": int(y1),
                        "xmax": int(x2), "ymax": int(y2),
                    })

        csv_path = DETECT_OUT_DIR / "detections.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()) if all_rows else [
                "image", "cls_id", "class", "conf", "xmin", "ymin", "xmax", "ymax"])
            writer.writeheader()
            writer.writerows(all_rows)

        n_defect = len({r["image"] for r in all_rows})
        cls_count = {n: sum(1 for r in all_rows if r["class"] == n) for n in CLASS_NAMES}
        detail = " | ".join(f"{n}: {c}" for n, c in cls_count.items())
        self.status.set(f"批量完成：{len(self.images)} 张 | 含缺陷 {n_defect} 张 | "
                        f"缺陷框 {len(all_rows)} 个（{detail}）")
        messagebox.showinfo(
            "批量检测完成",
            f"图片：{len(self.images)} 张（含缺陷 {n_defect} 张）\n"
            f"缺陷框：{len(all_rows)} 个（{detail}）\n\n"
            f"标注图：{ann_dir}\n记录文件：{csv_path}",
        )

    # ---------- 显示 ----------
    def _fit(self, im):
        w, h = im.size
        scale = min(MAX_W / w, MAX_H / h, 1.0)
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        return im

    def _show_image(self, canvas: tk.Canvas, path: Path, attr: str) -> None:
        from PIL import Image
        im = self._fit(Image.open(path).convert("RGB"))
        self._draw(canvas, im, attr)

    def _show_array(self, canvas: tk.Canvas, arr) -> None:
        from PIL import Image
        im = self._fit(Image.fromarray(arr[:, :, ::-1]))  # BGR -> RGB
        self._draw(canvas, im, "pred")

    def _draw(self, canvas: tk.Canvas, im, attr: str) -> None:
        from PIL import ImageTk
        photo = ImageTk.PhotoImage(im)
        setattr(self, f"_photo_{attr}", photo)
        canvas.delete("all")
        canvas.create_image(canvas.winfo_width() // 2, canvas.winfo_height() // 2,
                            image=photo, anchor=tk.CENTER)


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
