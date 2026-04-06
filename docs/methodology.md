# Bone Age Assessment — Preprocessing & DL Benchmarking

A study on how **image preprocessing quality affects deep learning model performance**
on pediatric bone age prediction from hand X-rays (RSNA 2017 dataset).

The project is structured as a 3 × 5 experiment matrix:

|  | ResNet-50 | EfficientNet-B3 | EfficientNet-V2 | ViT-B/16 | ConvNeXt-V2 |
|---|---|---|---|---|---|
| **Raw** | — | — | — | — | — |
| **Enhanced** (CLAHE) | — | — | — | — | — |
| **Cleaned** (Delabeled) | — | — | — | — | — |

> **Status:** Phase 1 (preprocessing pipeline) ✅ complete.  
> Phase 2 (model training & benchmarking) 🔄 in progress.  
> Experiment tracking on [Weights & Biases](https://wandb.ai/) — project `rsna-bone-age-matrix`.

***

## Why This Project

Hand X-rays in the RSNA dataset have significant variation in exposure,
positioning, and artifact density — ruler labels, text overlays, and detector
marks appear in random positions and sizes.

The core question is:

> *Does removing these artifacts and standardising contrast improve downstream
> model accuracy, and by how much?*

Most published bone age models train directly on raw X-rays. This project
isolates the preprocessing effect by holding model architecture, training code,
and evaluation split constant across all three data variants.

***

## Dataset

12,611 labelled pediatric hand X-rays. Labels: bone age in months (range 1–228).
Data hosted entirely on Kaggle — see [`data/README.md`](data/README.md).

| Stage | Source | Size |
|---|---|---|
| Raw | [kmader/rsna-bone-age](https://www.kaggle.com/datasets/kmader/rsna-bone-age) | 12,611 images |
| Enhanced | [ak7180979/preprocessed-rsna-boneage-xrays](https://www.kaggle.com/datasets/ak7180979/preprocessed-rsna-boneage-xrays) | 12,611 images |
| Cleaned (Delabeled) | [ak7180979/delabeled-rsna-boneage-xrays](https://www.kaggle.com/datasets/ak7180979/delabeled-rsna-boneage-xrays) | 12,611 images |

***

## Preprocessing Pipeline (Phase 1)

```
Raw X-ray (RSNA)
      │
      ▼
┌─────────────────────────────────┐
│  Step 1 – Enhancement           │
│  • Percentile normalisation     │  clips to [p1, p99], rescales to [0,255]— In Progress)

Five DL architectures will be trained on each of the three dataset variants
under identical conditions (same split, metrics, and training config):

| Architecture | Rationale |
|---|---|
| ResNet-50 | Baseline CNN with residual connections |
| EfficientNet-B3 | Compound-scaled, compute-efficient |
| EfficientNet-V2-M | Improved training speed and accuracy |
| ViT-B/16 | Vision Transformer; patch-based global attention |
| ConvNeXt-V2-Base | Modernised ConvNet; strong benchmark baseline |

**Evaluation metrics:** MAE (months), RMSE, % within ±6 months, % within ±12 months  
**Experiment tracking:** Weights & Biases (`rsna-bone-age-matrix` project)

---

## Expected Outcome

A 3×5 results matrix (dataset variant × architecture) comparing all metrics,
expected to surface:
1. Whether enhanced/cleaned inputs meaningfully reduce MAE.
2. Which architectures are most robust to image quality variation.
3. Whether FSCNN-based artifact removal is worth the computational overhead.