# YOLO 产品表面缺陷检测（划痕 / 凹坑）

基于 Ultralytics YOLO 的两类别表面缺陷检测项目：检测 **划痕（scratch）** 与 **凹坑（dent）**，
输出检测框可视化 + 准确率评估 + Excel 缺陷统计报表。

## 项目流程

```
数据准备 → YOLO 训练 → 分集合评估 → 批量推理(检测框) → Excel 统计报表
```

## 目录结构

```
├── src/
│   ├── config.py                                  # 路径/类别/默认参数统一配置
│   ├── data_prep/build_neu_yolo_dataset.py        # VOC→YOLO + 2类映射 + train/val/test 切分
│   ├── train.py                                   # 训练入口
│   ├── evaluate.py                                # train/val/test 分别评估 → 打印 + result.csv
│   ├── detect.py                                  # 批量检测 → 标注图 + detections.csv
│   ├── report.py                                  # detections.csv → Excel 报表
│   └── app.py                                     # 图形界面（选图 → 检测 → 看结果）
├── data/neu-det-2cls/                             # 生成的 YOLO 数据集(含 dataset.yaml)
├── runs/                                          # 训练/评估/检测/报表输出(运行时生成)
│   ├── train/<name>/                              # 训练产物(best.pt、曲线、results.csv)
│   ├── eval/<split>/ + result.csv                 # 评估产物与三套准确率
│   ├── detect/annotated/ + detections.csv         # 检测结果
│   └── report/缺陷统计报表.xlsx                   # Excel 报表
├── requirements.txt
└── README.md
```

## 数据集

演示/开发期使用公开数据集 **NEU-DET**（东北大学钢材表面缺陷库，200×200 灰度图，每类 300 张，共 6 类），
仅保留与本项目相关的两类并重命名：

| 原始类别                  | 本项目类别(id)     |
| --------------------- | ------------- |
| `scratches` 划痕        | `scratch` (0) |
| `pitted_surface` 点蚀/坑 | `dent` (1)    |

其余 4 类（crazing / inclusion / patches / rolled-in\_scale）会被过滤。后续如有真实产品图片，
整理为 YOLO 格式（images + labels + 与 `dataset.yaml` 同结构）后替换 `data/neu-det-2cls` 并重训即可。

### 数据切分

按类别分层随机切分（固定种子 42），默认 **train : val : test = 70% : 20% : 10%**：

| 集合 | 占比 | 张数 | 用途 |
| ---- | ---- | ---- | ---- |
| train | 70% | 420 | 训练（学参数） |
| val | 20% | 120 | 训练中每轮评估、选最优权重、早停 |
| test | 10% | 60 | 全程不参与训练/调参，最终"交卷"成绩 |

### 重新生成数据集（可选）

原始数据默认从 `%USERPROFILE%\Desktop\NEU-DET` 读取（可用环境变量 `RAW_NEU_DIR` 覆盖）：

```powershell
python -m src.data_prep.build_neu_yolo_dataset
# 自定义切分比例
python -m src.data_prep.build_neu_yolo_dataset --val-ratio 0.15 --test-ratio 0.15
```

## 环境准备

要求 Python 3.8+（推荐 3.10+）。**注意**：Windows 下请把虚拟环境建在短路径
（如 `C:\Users\<用户名>\.venv312`），不要建在过深的项目目录内，否则安装 torch 会触发
260 字符路径上限报错。

### 1. 创建虚拟环境

```powershell
py -3.12 -m venv C:\Users\<用户名>\.venv312
```

### 2. 安装依赖

```powershell
C:\Users\<用户名>\.venv312\Scripts\python -m pip install --upgrade pip
C:\Users\<用户名>\.venv312\Scripts\python -m pip install -r requirements.txt
```

> `requirements.txt` 默认从 PyPI 解析，装出的是 **CPU 版 torch**；只有 NVIDIA GPU 机器需要步骤 3。

### 3.（GPU 机器可选）换装 CUDA 版 torch

```powershell
C:\Users\<用户名>\.venv312\Scripts\python -m pip uninstall -y torch torchvision
C:\Users\<用户名>\.venv312\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# 验证
C:\Users\<用户名>\.venv312\Scripts\python -c "import torch;print(torch.cuda.is_available())"
```

## 训练

无 GPU 时自动退化为 CPU（建议减小尺寸/轮次，如 `--imgsz 384 --epochs 80`）：

```powershell
cd <项目目录>
python -m src.train                                                    # 默认参数训练(yolov8n)
python -m src.train --model yolov8s.pt --epochs 200 --batch 16         # 自定义起点模型/超参
python -m src.train --model runs\train\neu2cls\weights\best.pt         # 断点续训/微调
```

训练日志与权重输出在 `runs/train/<name>/`，最优权重为 `runs/train/<name>/weights/best.pt`。

> 首次运行会下载预训练权重 `yolov8n.pt`。若 GitHub 无法访问，可先手动下载放到项目根目录或 `src/` 下，
> Ultralytics 会直接使用本地文件（可用镜像：`https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8n.pt`）。

## 评估准确率（train / val / test 各自）

```powershell
python -m src.evaluate                         # 三个集合都评，结果存 runs\eval\result.csv
python -m src.evaluate --splits test           # 只评测试集
```

输出表格（Precision / Recall / mAP50 / mAP50-95），并写入 `runs/eval/result.csv`；
各集合的混淆矩阵、P-R 曲线等图在 `runs/eval/<split>/`。

## 批量检测

```powershell
python -m src.detect --weights runs\train\neu2cls\weights\best.pt --source data\neu-det-2cls\images\test --device 0
python -m src.detect --weights <best.pt> --source <图片目录|单张图> --conf 0.25 --out runs\detect
```

输出：

- `runs/detect/annotated/` —— 带检测框(类别+置信度)的标注图
- `runs/detect/detections.csv` —— 每个缺陷框的结构化记录（供报表使用）

## 图形界面（推荐日常使用）

```powershell
python -m src.app
```

窗口内可：选择图片/文件夹 → 拖动置信度阈值 → 检测当前图（左原图 / 右带框图 + 结果表）→
上一张/下一张浏览 → 批量检测并保存（标注图 + detections.csv）。

## 生成 Excel 报表

```powershell
python -m src.report --detections runs\detect\detections.csv
python -m src.report --detections <csv> --out <报表.xlsx> --ok-max-defects 0
```

报表包含 3 个 Sheet：

1. **缺陷明细**：每个缺陷框一行（图片/类别/置信度/坐标/框尺寸）
2. **图片判定**：每张图各类缺陷数量与合格/不合格判定（默认存在任意缺陷即不合格，可用 `--ok-max-defects` 放宽）
3. **汇总统计**：总图数、缺陷图占比、各类检出总数、不合格率等

> 注意：报表按"检出的缺陷框"聚合，**没有检出任何缺陷的图片不会出现在报表中**。

## 常见问题

- **安装 torch 报** **`OSError: No such file ...`**：路径过长导致，把 venv 放到短路径重装。

- **`torch.cuda.is_available()`** **为 False**：未装 CUDA 版 torch，或机器无 NVIDIA GPU，按上面步骤 3 处理。

- **权重下载失败 / SSL 报错**：本地放好预训练权重即可离线运行；或用镜像
  `https://hf-mirror.com/Ultralytics/YOLOv8/resolve/main/<模型名>.pt` 手动下载。

- **划痕是细长目标、凹坑偏小目标**：如真实数据上效果不佳，可提高 `imgsz`（如 1024）、增大 `epochs` 或换 `yolov8m/s`。
