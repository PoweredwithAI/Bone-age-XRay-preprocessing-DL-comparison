# Bone Age Assessment — Preprocessing & DL Benchmarking

A study on how **image preprocessing quality affects deep learning model performance** on pediatric bone age prediction from hand X-rays (RSNA 2017 dataset).

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

# Methodology — Bone Age X-Ray Preprocessing Pipeline

This document describes the full preprocessing methodology applied to the RSNA Pediatric Bone Age dataset prior to deep learning model training. The pipeline is divided into two stages: **image enhancement** (Stage 1) and **artefact and label removal** (Stage 2), with a validation step in between using DICE scoring.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Dataset Overview](#2-dataset-overview)
3. [Why Preprocessing Matters](#3-why-preprocessing-matters)
4. [Stage 1 — Image Enhancement](#4-stage-1--image-enhancement)
5. [Stage 2 — Artefact and Label Removal via FSCNN](#5-stage-2--artefact-and-label-removal-via-fscnn)
6. [Stage 2b — DICE Validation](#6-stage-2b--dice-validation)
7. [Three Dataset Variants for Downstream Training](#7-three-dataset-variants-for-downstream-training)
8. [Common X-Ray Artefact Sources](#8-common-x-ray-artefact-sources)
9. [Key Design Decisions](#9-key-design-decisions)
10. [References](#10-references)

---

## 1. Problem Statement

Bone age assessment from hand X-rays is a clinical task where a radiologist estimates skeletal maturity from the appearance of growth plates and bone structure. Automating this with deep learning requires clean, consistent images. The raw RSNA dataset presents several challenges:

- **Variable exposure and contrast** — images range from severely underexposed to overexposed, making pixel intensity non-comparable across samples.
- **Non-uniform hand positioning** — hands are rotated, cropped, or off-centre across images.
- **Labels and external artefacts** — text overlays, ruler scales, bounding boxes, and other non-anatomical elements are embedded directly in the image pixels in unpredictable positions.
- **Variable image sizes** — the dataset contains images at different resolutions, requiring normalisation before batching.

The hypothesis driving this preprocessing work is:

> *Removing non-anatomical information from training images will improve model generalisation and reduce reliance on spurious correlations between artefact position and bone age label.*

---

## 2. Dataset Overview

| Property | Value |
|---|---|
| Dataset | RSNA Pediatric Bone Age Challenge 2017 |
| Total training images | 12,611 |
| Image type | Hand X-ray (greyscale PNG) |
| Label | Bone age in months (1–228) |
| Gender split | 6,833 Male / 5,778 Female |
| Source | [Kaggle — kmader/rsna-bone-age](https://www.kaggle.com/datasets/kmader/rsna-bone-age) |

---

## 3. Why Preprocessing Matters

X-ray image preprocessing identifies and removes artefacts — such as noise, grid lines, and detector defects — using techniques like bad pixel correction, flat-field correction, median filtering, and Fourier spectral filtering.
These methods enhance image quality and ensure diagnostic accuracy by removing unwanted distortions before clinical analysis.

### Key artefact identification and removal techniques

| Technique | Description |
|---|---|
| **Bad Pixel Correction** | Identifies dead or unresponsive pixels on digital flat panels and corrects them by averaging surrounding pixel values |
| **Flat Field Correction** | Corrects for uneven X-ray exposure and detector sensitivity variations using gain and offset images (dark image subtraction) |
| **Noise Reduction Filtering** | Employs median filters for salt-and-pepper noise and Gaussian filters for speckle reduction |
| **Grid Line Suppression** | Uses Fourier domain filtering or wavelet transforms to remove stationary grid lines without removing diagnostic content |
| **Artefact Removal Algorithms** | Tackles streaks, bright-burn artefacts from overexposure, or ring artefacts in CT by interpolating affected areas |

### General X-ray preprocessing steps

1. **Normalisation / Calibration (Demonstrated)** — Offset-corrected first correction image removes afterglow.
2. **Smoothing (Planned)** — Wavelet thresholding or Butterworth low-pass filters reduce random noise.
3. **Enhancement (Demonstrated)** — CLAHE improves visibility of fine anatomical detail after artefact removal.
4. **AI Integration (Demonstrated)** — Machine learning algorithms are increasingly used for automatic identification and removal of complex, non-linear artefacts that rule-based methods cannot handle.

---

## 4. Stage 1 — Image Enhancement

**Script:** `src/preprocessing.py`
**Config:** `configs/preprocessing.yaml` → `clahe` and `normalisation` blocks
**Output:** Enhanced image dataset + `RSNA_boneage_enhanced_training_dataset.csv`

### 4.1 Step 1 — Percentile Normalisation

Applied to **every image** before any enhancement decision is made.

```

p_low  = 1st percentile pixel intensity
p_high = 99th percentile pixel intensity
clipped = clip(image, p_low, p_high)
output  = rescale(clipped) →  uint8

```

**Why percentile and not min-max?**
Min-max normalisation is distorted by even a single extreme outlier pixel (e.g. a burned-in label at 255 or a dead pixel at 0). Clipping to the 1st–99th percentile removes these outliers before rescaling, producing a stable intensity range across all images regardless of exposure variation.

### 4.2 Step 2 — Dullness Detection

After normalisation, each image is checked for underexposure:

```python
is_dull = mean(normalised_image) < 50
```

A mean pixel intensity below 50 (on a 0–255 scale) indicates an underexposed image that lacks sufficient contrast for reliable feature extraction. Only these images proceed to CLAHE.

**Why gate CLAHE on dullness?**
Applying CLAHE universally to well-exposed images risks amplifying noise and creating artificial texture in regions that are already diagnostically clear. Selective application ensures CLAHE improves images that need it without degrading those that do not.

### 4.3 Step 3 — CLAHE (Contrast Limited Adaptive Histogram Equalisation)

Applied **only to dull images** (or all images if `force_enhance=True`).


| Parameter | Value | Rationale |
| :-- | :-- | :-- |
| `clip_limit` | 2.0 | Caps contrast amplification per tile — prevents noise amplification in uniform regions |
| `tile_grid_size` | (8, 8) | 64 local tiles — fine enough to handle local variation, coarse enough to avoid tiling artefacts |
| Implementation | `skimage.exposure.equalize_adapthist` | Float [0,1] input; result rescaled back to uint8 |

**Why CLAHE over global Histogram Equalisation (HE)?**

Global HE computes a single histogram transformation across the entire image. In X-rays with strong local variation (bright bone, dark soft tissue), this causes noise amplification in already-bright regions and loss of detail in darker areas. CLAHE operates on local tiles with a contrast cap, preserving local contrast without these artefacts. Multiple studies confirm CLAHE
outperforms HE and gamma correction for vertebral bone segmentation tasks.

**Why CLAHE over Gamma Correction?**

Gamma correction applies a global power-law transformation that uniformly brightens or darkens the entire image. It cannot adapt to local regions and provides no mechanism to cap contrast amplification.

### 4.4 Idempotent build-or-load pattern

The enhancement pipeline uses a toggle flag (`USING_KAGGLE_DATASET`) to switch between first-run processing and fast reuse of pre-processed images:

```
USING_KAGGLE_DATASET = False → Process all images, save to /kaggle/working
                                Download and upload as Kaggle dataset
USING_KAGGLE_DATASET = True  → Load CSV and image paths directly from
                                uploaded Kaggle dataset — no reprocessing
```

This prevents accidentally re-running expensive processing on every notebook session restart and is a standard pattern for reproducible Kaggle workflows.

---

## 5. Stage 2 — Artefact and Label Removal via FSCNN

**Scripts:** `src/fscnn_module.py` (model + training), `src/artifact_removal.py` (inference + cleaning)
**Config:** `configs/preprocessing.yaml` → `fscnn_training` and `fscnn_inference` blocks
**Output:** Cleaned image dataset + `RSNA_boneage_cleaned_training_dataset.csv`

### 5.1 Why FSCNN for segmentation?

Rather than using classical thresholding or edge detection to separate the hand from background artefacts, this pipeline uses **FastSurferCNN** — a U-Net-style encoder-decoder architecture — to predict a pixel-wise binary mask of the hand region. The mask is then used to zero out all non-hand pixels (labels, rulers, text, background).

**Why learning-based segmentation over thresholding?**


| Approach | Limitation |
| :-- | :-- |
| Global thresholding | Fails when label intensity overlaps with bone intensity |
| Edge detection | Sensitive to noise; produces fragmented boundaries |
| Classical morphology | Cannot handle arbitrary label positions and shapes |
| **FSCNN (learned)** | Generalises across all label positions, hand sizes, and orientations |

### 5.2 Manual masks and training data

Ground-truth hand masks were obtained from:

> Rassmann, S., Keller, A., Skaf, K. et al. *Deeplasia: deep learning for
> bone age assessment validated on skeletal dysplasias.*
> Pediatr Radiol 54, 82–95 (2024).
> [https://doi.org/10.1007/s00247-023-05789-1](https://doi.org/10.1007/s00247-023-05789-1)

Masks were downloaded from [Zenodo (records/7611677)](https://zenodo.org/records/7611677).
They were obtained manually using thresholding and edge detection, and all masks were quality-checked and corrected by the original authors.

**RGBA pair construction** (handled inside `src/fscnn_module.py`)**:**
Each enhanced X-ray with a corresponding manual mask is combined into a
4-channel RGBA image for `MaskDataSet` compatibility:

```
Channels 0–2 (RGB) : Enhanced greyscale X-ray converted to 3-channel
Channel  3   (A)   : Binary hand mask (0 = background, 255 = hand)
```


### 5.3 Model architecture

Defined in `src/fscnn_module.py` as `FSCNNLightning` — a PyTorch Lightning
wrapper around `FastSurferCNN`.


| Property | Value |
| :-- | :-- |
| Architecture | FastSurferCNN (U-Net encoder-decoder) |
| Input channels | 1 (greyscale) |
| Output classes | 2 (hand foreground, background) |
| Kernel size | (5, 5) |
| Filters | 64 |
| Loss | CombinedLoss = DICE + Cross-Entropy (equal weights) |
| Optimiser | Adam, lr = 1e-3 |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Max epochs | 40 |
| Input size | 512 × 512 |
| Train / val split | 90 / 10 |

**Why CombinedLoss (DICE + CE)?**

- **DICE loss** handles the class imbalance between hand foreground and background; it optimises directly for overlap rather than per-pixel accuracy.
- **Cross-Entropy loss** provides stable per-pixel gradients, especially early in training when the DICE denominator is large.
- Equal weighting (1.0 + 1.0) balances both objectives throughout training.

**Why ReduceLROnPlateau?**
Avoids manual LR scheduling. Automatically halves the learning rate when `val_loss` does not improve for 3 consecutive epochs, allowing training to continue productively without hyperparameter tuning between runs.

### 5.4 Augmentation strategy`

The upstream `MaskModule` augmentation defaults were replaced via monkey-patching inside `src/fscnn_module.py` (no repo fork required) to add medical imaging-relevant transforms:


| Augmentation | Purpose |
| :-- | :-- |
| VerticalFlip / HorizontalFlip (p=0.5) | Handles left/right hand variation |
| RandomRotate90 (p=0.5) | Handles non-standard hand orientations |
| RandomBrightnessContrast (±0.3) | Simulates exposure variation across scanners |
| RandomResizedCrop (scale 0.8–1.0) | Teaches model to handle partial hands near image boundaries |
| GaussNoise (std 0.04–0.16) | Simulates real-world DICOM image noise |
| ImageCompression (quality 25–80) | Simulates DICOM compression and lossy storage |
| alex_aug (random black rectangles) | Simulates label occlusion during training |

**alex_aug patch:**
The upstream implementation contained a bug where `x0 == x1` or `y0 == y1` produced a zero-area rectangle, raising a `Pillow ValueError`. The patched version guarantees at least 1-pixel width and height.

### 5.5 Inference and artefact removal

Handled by `src/artifact_removal.py`.

**Predict mask:**

1. Read enhanced X-ray as greyscale uint8
2. Resize to 512 × 512, normalise to [0, 1]
3. Forward pass through FSCNN → logits → softmax → foreground probability map
4. Threshold at 0.5 → binary mask at model resolution
5. Resize binary mask back to original image dimensions (nearest-neighbour)

**Remove artefacts — two strategies:**


| Strategy | Method | When to use |
| :-- | :-- | :-- |
| `blackout` | `cv2.bitwise_and` — zero out all non-hand pixels | Default; fast; lossless within hand region |
| `inpaint` | TELEA inpainting — fill non-hand region with surrounding pixel context | When downstream models are sensitive to hard black borders |


---

## 6. Stage 2b — DICE Validation

**Script:** `src/dice_eval.py`

Before the cleaned dataset is used for downstream model training, the quality of FSCNN-predicted masks is validated against the manual ground-truth masks using the DICE similarity coefficient.

```
DICE = (2 × |pred ∩ gt|) / (|pred| + |gt|)

1.00  → perfect overlap
≥0.95 → clinically acceptable for segmentation
<0.90 → mask quality may degrade artefact removal
```

**Efficiency design:**
Rather than calling `os.path.exists()` per row across all 12,611 images, the evaluation loop is pre-filtered using a single `os.listdir()` scan of the mask directory. This reduces filesystem calls from 12,611 → 1 and loop iterations to only the ~9,000 images that have a manual mask.

The original authors report **DICE ≈ 0.99** on this dataset using the FSCNN approach (Rassmann et al. 2024).

---

## 7. Three Dataset Variants for Downstream Training

The preprocessing pipeline produces three distinct datasets that will be used in Phase 2 to benchmark deep learning model performance:


| Variant | Description | Preprocessing Applied |
| :-- | :-- | :-- |
| **Raw** | Original RSNA images | None |
| **Enhanced** | Contrast-improved images | Percentile normalisation + selective CLAHE |
| **Cleaned** | Artefact and label removed | All of the above + FSCNN mask → blackout |

The central research question for Phase 2 is:

> *Which preprocessing strategy — none, enhancement only, or full artefact removal — produces the most accurate and generalisable bone age prediction model?*

---

## 8. Common X-Ray Artefact Sources

Understanding what artefacts exist in the RSNA dataset informs both the choice of preprocessing technique and the design of the FSCNN training augmentations.

### Hardware defects
- **Dead / stuck pixels** — individual detector elements that permanently read zero or maximum intensity regardless of actual exposure.
- **Damaged flat panels** — sections of the detector array that are miscalibrated, producing banding or patchy intensity regions.
- **Dust and debris** — particles on the detector surface that cast shadows onto the image.

### Image receptor artefacts
- **Computed Radiography (CR) artefacts** — scratches or damage to phosphor imaging plates that appear as bright or dark streaks.
- **Direct Radiography (DR) artefacts** — detector pixel dropout, ghosting from previous exposures (afterglow), and gain map miscalibration.
- **Grid lines** — stationary anti-scatter grids produce periodic line patterns that can interfere with texture-sensitive models.

### Patient and external object artefacts
- **Jewellery** — rings, bracelets, and watches appear as bright high-density objects that can partially occlude bone structures.
- **Clothing with metal** — bra underwires and belt buckles produce strong attenuation artefacts near the field of view.
- **Hair** — dense hair over the hand can create soft-tissue attenuation that is misinterpreted as pathology by a model.

### Imaging system artefacts specific to the RSNA dataset
- **Embedded text labels** — patient ID, age, date, and laterality markers are burned directly into image pixels in positions that vary across institutions and scanners.
- **Ruler and scale markers** — radiopaque rulers placed at the image boundary appear as bright linear structures.
- **Bounding box overlays** — some images contain drawn annotations or crop markers from the original clinical workflow.
- **Brightness and contrast variation** — images from different hospitals and scanner manufacturers have systematically different baseline intensity distributions, producing a non-i.i.d. dataset.

These artefacts are the primary motivation for the two-stage preprocessing pipeline: Stage 1 corrects global intensity variation, and Stage 2 removes spatially localised non-anatomical content.

---

## 9. Key Design Decisions

A summary of the most consequential technical choices made in this pipeline and the rationale behind each.

| Decision | Chosen approach | Alternatives considered | Reason |
|---|---|---|---|
| **Intensity normalisation** | Percentile (1st–99th) | Min-max, z-score | Robust to extreme outlier pixels from burned-in labels |
| **Contrast enhancement** | CLAHE | Global HE, Gamma correction | Local adaptivity + contrast cap prevents noise amplification |
| **CLAHE gating** | Only on dull images (mean < 50) | Apply to all | Avoids degrading well-exposed images |
| **Segmentation method** | FSCNN (learned) | Otsu threshold, Canny edge, morphology | Generalises to arbitrary label positions and hand orientations |
| **Loss function** | DICE + Cross-Entropy (equal weights) | DICE only, CE only, Focal loss | DICE handles class imbalance; CE stabilises early training gradients |
| **LR scheduling** | ReduceLROnPlateau | StepLR, CosineAnnealing, fixed LR | Adaptive — no manual schedule tuning needed |
| **Artefact removal** | Blackout (bitwise AND with mask) | TELEA inpainting, cropping | Fast, lossless within hand region, no hallucinated pixels |
| **Augmentation delivery** | Monkey-patch upstream MaskModule | Fork the repo | Keeps dependency clean and traceable without maintaining a fork |
| **alex_aug fix** | Clamp x1, y1 to guarantee 1px | Skip augmentation | Upstream bug caused Pillow ValueError on zero-area rectangles |
| **DICE evaluation scope** | Pre-filter to masked rows via listdir | Check exists() per row | Single filesystem scan vs 12,611 individual calls |
| **Dataset variants** | Raw / Enhanced / Cleaned | Single preprocessed dataset | Enables controlled ablation of preprocessing contribution to model accuracy |

---

## 10. References

### Dataset and masks

1. **RSNA Pediatric Bone Age Challenge Dataset**
   Halabi SS, Prevedello LM, Kalpathy-Cramer J, et al.
   *The RSNA Pediatric Bone Age Machine Learning Challenge.*
   Radiology 2019; 290(2): 498–503.
   [Kaggle dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age)

2. **Manual hand segmentation masks and FSCNN**
   Rassmann S, Keller A, Skaf K, et al.
   *Deeplasia: deep learning for bone age assessment validated on skeletal
   dysplasias.*
   Pediatr Radiol 54, 82–95 (2024).
   [https://doi.org/10.1007/s00247-023-05789-1](https://doi.org/10.1007/s00247-023-05789-1)
   Masks available at [Zenodo records/7611677](https://zenodo.org/records/7611677)

3. **FSCNN segmentation module**
   [@aimi-bonn](https://github.com/aimi-bonn) /
   [hand-segmentation](https://github.com/aimi-bonn/hand-segmentation/tree/master/FSCNN/lib)

### Preprocessing and enhancement

4. **Image preprocessing techniques primer**
   Fiveable — Image Preprocessing Techniques
   [https://fiveable.me/lists/image-preprocessing-techniques](https://fiveable.me/lists/image-preprocessing-techniques)

5. **CLAHE vs HE vs Gamma Correction for X-ray**
   *An analysis of x-ray image enhancement methods for vertebral bone
   segmentation*
   ResearchGate, 2014.
   [https://www.researchgate.net/publication/269309071](https://www.researchgate.net/publication/269309071_An_analysis_of_x-ray_image_enhancement_methods_for_vertebral_bone_segmentation)

6. **scikit-image CLAHE implementation**
   `skimage.exposure.equalize_adapthist`
   [https://scikit-image.org/docs/stable/api/skimage.exposure.html](https://scikit-image.org/docs/stable/api/skimage.exposure.html)

### Artefact removal

7. **OpenCV TELEA inpainting**
   Telea A. *An Image Inpainting Technique Based on the Fast Marching
   Method.* Journal of Graphics Tools, 9(1), 2004.
   [OpenCV inpainting docs](https://docs.opencv.org/4.x/df/d3d/tutorial_py_inpainting.html)

---

*This document covers Phase 1 (preprocessing) only.
Phase 2 methodology — model architecture selection, training protocol, and cross-dataset benchmarking — will be added upon completion of comparative experiments.*