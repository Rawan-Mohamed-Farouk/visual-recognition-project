# 🏥 Medical PPE Detection

A YOLOv8-based object detection system for identifying **Personal Protective Equipment (PPE)** worn by medical staff. The project fine-tunes and compares two YOLOv8 configurations on a custom augmented dataset of 5 PPE classes.

## 📌 Overview

Medical environments require strict PPE compliance. This project trains an object detector that can automatically identify whether healthcare workers are wearing the correct protective equipment — a step toward automated safety monitoring in clinical settings.

The notebook runs end-to-end on Google Colab with a T4 GPU and covers dataset preparation, model training, evaluation, and comparison.

---

## 🎯 Detected Classes

| ID | Class        | Images in Dataset |
|----|--------------|-------------------|
| 0  | `cap`        | 564               |
| 1  | `face_mask`  | 378               |
| 2  | `gloves`     | 498               |
| 3  | `gown`       | 552               |
| 4  | `shoe_cover` | 600               |

**Total: ~2,592 images** (augmented dataset stored in Google Drive)

---

## 🗂️ Dataset Structure

The source dataset is an augmented image-classification-style folder tree stored in Google Drive. The notebook converts it to YOLO detection format with a full-image bounding box and splits it reproducibly:

```
augmented_dataset/
├── cap/
├── face_mask/
├── gloves/
├── gown/
└── shoe_cover/
```

Converted to:
```
ppe_yolo_dataset/
├── images/
│   ├── train/   (70%)
│   ├── val/     (20%)
│   └── test/    (10%)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Split sizes (approx.): **1,815 train · 518 val · 259 test** — with zero overlap between splits confirmed.

---

## 🔬 Model Configurations

Two YOLOv8 configurations are trained and compared:

| Parameter   | Config 1 — Baseline         | Config 2 — Tuned            |
|-------------|-----------------------------|-----------------------------|
| Model       | `yolov8n.pt` (nano, 3M params) | `yolov8s.pt` (small, 11M params) |
| Epochs      | 30                          | 50                          |
| Batch size  | 16                          | 16                          |
| Image size  | 640×640                     | 640×640                     |
| Learning rate (lr0) | 0.01               | 0.005                       |
| LR final (lrf)     | 0.01                | 0.001                       |
| Optimizer   | SGD                         | AdamW                       |
| Dropout     | 0.0                         | 0.1                         |

---

## 📊 Results

### Config 1 — YOLOv8n Baseline (30 epochs)

| Metric     | Value  |
|------------|--------|
| mAP@50     | ~0.993 |
| mAP@50-95  | ~0.992 |
| Precision  | ~0.98  |
| Recall     | ~0.964 |

### Config 2 — YOLOv8s Tuned (50 epochs)

| Metric     | Value  |
|------------|--------|
| mAP@50     | **0.995** |
| mAP@50-95  | **0.995** |
| Precision  | **0.996** |
| Recall     | **0.995** |

**Per-class breakdown (Config 2 best model):**

| Class       | Precision | Recall | mAP@50 | mAP@50-95 |
|-------------|-----------|--------|--------|-----------|
| cap         | 1.000     | 0.995  | 0.995  | 0.995     |
| face_mask   | 0.983     | 1.000  | 0.995  | 0.995     |
| gloves      | 1.000     | 0.981  | 0.995  | 0.995     |
| gown        | 0.999     | 1.000  | 0.995  | 0.995     |
| shoe_cover  | 0.998     | 1.000  | 0.995  | 0.995     |

> **Winner: Config 2 (YOLOv8s Tuned)** — achieves near-perfect detection across all 5 classes. Inference speed: ~4.3 ms per image on Tesla T4 GPU.

> ⚠️ Note: A pretrained COCO model tested without fine-tuning showed near-zero performance on PPE classes, confirming that domain-specific fine-tuning was essential.

---

## 🚀 Getting Started

### Requirements

```bash
pip install ultralytics scikit-learn matplotlib seaborn pandas
```

Or open directly in Google Colab (recommended) — all dependencies are installed in the first cell.

### Dataset

Place your augmented PPE dataset in Google Drive at:
```
MyDrive/augmented_dataset/
```
with one subdirectory per class (`cap`, `face_mask`, `gloves`, `gown`, `shoe_cover`).

### Running the Notebook

1. Open the notebook in [Google Colab](https://colab.research.google.com/github/Anton3090/Medical-detection/blob/main/Medical-detection.ipynb)
2. Set runtime to **GPU** (T4 recommended)
3. Mount Google Drive and ensure the dataset path is correct
4. Run all cells sequentially

---

## 📁 Notebook Structure

| Cell | Description |
|------|-------------|
| 1    | Mount Google Drive & install dependencies |
| 2    | Import libraries |
| 3    | Configuration — class names, paths, hyperparameters |
| 4    | Convert classification dataset → YOLO detection format (no data leakage) |
| 5    | Dataset statistics & split verification |
| 6    | Generate `data.yaml` for YOLO |
| 7    | Define two hyperparameter configurations |
| 8    | Train Config 1 (YOLOv8n Baseline) |
| 9    | Train Config 2 (YOLOv8s Tuned) |
| 10   | Evaluate both models on the test set |
| 11   | Compare models & select winner |
| 12   | Visualize training curves |
| 13   | Baseline comparison (pretrained COCO vs fine-tuned) |
| 14   | Display confusion matrix |
| 15   | Verify no data leakage between splits |

---

## 🛠️ Tech Stack

- **Framework:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Platform:** Google Colab (CUDA, Tesla T4 GPU)
- **Language:** Python 3.12
- **Libraries:** PyTorch, OpenCV, scikit-learn, matplotlib, seaborn, pandas, Pillow

---

## 📄 License

This project is open-source. Feel free to use and adapt for research or educational purposes.
