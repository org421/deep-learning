# ModelTrain -- Scientific Image Forgery Detection

This repository contains three model architectures for scientific image copy-move forgery segmentation:

| Model | Directory | Backbone | Decoder |
|---|---|---|---|
| **DinoV2-UNET** | `src_dinov2/` | DinoV2 ViT-B/14 (via `torch.hub`) | UNET |
| **SegFormer** | `src_SegFormer/` | MiT (Mix Transformer) B0-B5 | All-MLP |
| **UNET-ResNet50** | `src_UNET_RESNET50/` | ResNet50 (ImageNet pretrained) | UNET (`segmentation-models-pytorch`) |
| **ConvNeXt-UNET** | `src_convnext/` | ConvNeXt-Base (ImageNet-22k→1k via `timm`) | UNET (GroupNorm) |
| **SAM (Segment Anything)** | `src_sam/` | ViT-H / ViT-L / ViT-B (pretrained, Meta AI) | Prompt-based mask decoder |
| **CMSegNet** | `src_cmsegnet/` | Dual-branch CNN (Context + Mask streams) | Custom segmentation decoder |
| **DinoV3-UNET** | `src_dinov3/` | DINOv3 ViT  | UNET-style segmentation head |

All three share the same dataset format, evaluation metric (Kaggle oF1), and a similar training/inference pipeline.

---

## 1. Installation

### Prerequisites

- Python 3.10+
- CUDA 12.x (compatible with the RTX 5070 Ti)

### Steps

```bash
# Install dependencies
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA support may require a specific install command depending on your CUDA version. Check https://pytorch.org/get-started/locally/ and use the appropriate command, e.g.:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
> ```

---

## 2. Dataset Structure

All three models expect data in the following layout (configured via `data_root` in each `configs/config.yaml`):

```
data/
├── train/
│   ├── images/
│   │   ├── forged/        
│   │   └── authentic/     
│   └── masks/             
├── val/
│   ├── images/
│   │   ├── forged/
│   │   └── authentic/
│   └── masks/
└── test/
    ├── images/
    │   ├── forged/
    │   └── authentic/
    └── masks/
```


---

## 3. Training

Each model is trained by running `main.py --mode train` inside its source directory. Hyperparameters are read from `configs/config.yaml`.

### Models

```
cd src_<model>
python main.py --mode train
```

Exemple:

```bash
cd src_dinov2
python main.py --mode train
```

---

## 4. Inference / Evaluation

### Quick Evaluation (all models)

```bash
cd src_<model>
python main.py --mode inference \
    --checkpoint <path/to/model_best_oF1.pth> \
    --split test \
    --output results.json
```

### Full Inference Pipeline (all models)

Each model also includes `main_inference.py`, which runs a grid search over (threshold, min_area) on the validation set and evaluates on the test set with the best parameters:

```bash
cd src_<model>
python main_inference.py --work_dir ./work_dir --data_root ../data
```

Options:
- `--visualize_image <filename>` : generate overlay visualizations for a specific image.

### Outputs

| File | Content |
|---|---|
| `inference_report_<timestamp>.txt` | Human-readable report with grid search table and per-category metrics |
| `inference_results_<timestamp>.json` | Machine-readable results |
| `visualization_*.png` | Prediction overlays vs. ground truth |

### Metrics

- **oF1** (Optimal F1) -- primary Kaggle competition metric (instance-level matching with Hungarian algorithm)
- **Forged oF1** -- oF1 restricted to forged images
- Pixel-level: Dice, IoU, Precision, Recall

---

## 5. Trained Weights

During training, checkpoints are saved inside the experiment's `work_dir`:

```
<work_dir>/
└── <experiment_name>-<timestamp>/
    ├── model_best_loss.pth          # Best validation loss
    ├── model_best_oF1.pth           # Best oF1 score
    ├── model_best_loss_ema.pth      # EMA variant
    ├── model_best_oF1_ema.pth       # EMA variant
    ├── model_latest.pth             # Latest epoch
    ├── config.yaml                  # Config snapshot
    ├── metrics_history.json         # Per-epoch metrics
    └── training_curves.png          # Loss / metric plots
```

The default `work_dir` for each model is configured in its `configs/config.yaml`. Look for the most recent timestamped subdirectory.

---

They can be downloaded from the following Google Drive folder because they took lot of memory:  
https://drive.google.com/drive/folders/1mkST-0apNJf8yQ7Pm7903QvWIXv6yscF

After downloading, place the desired `.pth` checkpoint file in work_dir/ folder.
---

## 6. Hardware / Environment (Ozan Gunes Setup)

| Component | Specification |
|---|---|
| **GPU** | NVIDIA RTX 5070 Ti -- 16 GB VRAM |
| **RAM** | 32 GB DDR5 |
| **CPU** | AMD Ryzen 7 7700 (8 cores / 16 threads) |
| **OS** | Windows 11 |
| **IDE** | Visual Studio Code |
| **Python** | 3.10+ |
| **CUDA** | 12.x |
| **Framework** | PyTorch 2.x with AMP (mixed precision) |

