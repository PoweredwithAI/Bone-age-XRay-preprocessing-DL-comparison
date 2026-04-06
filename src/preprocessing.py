"""
preprocessing.py
----------------
Step 1: Image enhancement pipeline for RSNA Bone Age X-rays.

Functions:
  - percentile_normalize   : Clips and rescales pixel intensities to [0, 255]
  - apply_clahe            : Applies CLAHE using scikit-image
  - is_dull                : Flags underexposed images by mean pixel intensity
  - build_or_load_enhanced_df : Orchestrates the full enhancement run or loads
                                a pre-built Kaggle dataset
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage import exposure

# ─── CLAHE hyper-parameters ──────────────────────────────────────────────────
DULL_THRESHOLD = 50    # Images whose mean pixel value < this get CLAHE applied
CLAHE_CLIP     = 2.0   # Clip limit for adaptive histogram equalisation
CLAHE_TILE     = (8, 8)  # Tile grid size for CLAHE

# ─── Kaggle dataset paths (read-only input; set for reuse after first run) ────
KAGGLE_ENHANCED_DIR = (
    "/kaggle/input/datasets/ak7180979/preprocessed-rsna-boneage-xrays"
    "/RSNAboneageenhanced/RSNAboneageenhancedtrainingdataset"
)
CSV_KAGGLE_PATH = (
    "/kaggle/input/datasets/ak7180979/preprocessed-rsna-boneage-xrays"
    "/RSNAboneageenhancedtrainingdataset.csv"
)

# ─── Local working paths (used on first run before upload to Kaggle) ──────────
LOCAL_ENHANCED_DIR = "/kaggle/working/RSNAboneageenhancedtrainingdataset"
CSV_SAVE_PATH      = (
    "/kaggle/working/RSNAboneageenhancedtrainingdataset"
    "/RSNAboneageenhancedtrainingdataset.csv"
)

# ─── Toggle: True → load from Kaggle dataset; False → run enhancement ─────────
USING_KAGGLE_DATASET = True
ENHANCED_DIR = KAGGLE_ENHANCED_DIR if USING_KAGGLE_DATASET else LOCAL_ENHANCED_DIR
CSV_PATH     = CSV_KAGGLE_PATH     if USING_KAGGLE_DATASET else CSV_SAVE_PATH


# ─── Core image functions ─────────────────────────────────────────────────────

def percentile_normalize(img_gray: np.ndarray, low: int = 1, high: int = 99) -> np.ndarray:
    """
    Clip pixel intensities to [low, high] percentile then rescale to [0, 255].

    Removes extreme over/underexposed pixels before CLAHE so the adaptive
    equalisation operates on a clean intensity range.

    Args:
        img_gray : Grayscale image array (uint8 or float).
        low      : Lower percentile for clipping (default 1).
        high     : Upper percentile for clipping (default 99).

    Returns:
        Normalised uint8 image in [0, 255].
    """
    p_low  = np.percentile(img_gray, low)
    p_high = np.percentile(img_gray, high)
    clipped = np.clip(img_gray, p_low, p_high)
    return cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def apply_clahe(
    img_gray: np.ndarray,
    clip_limit: float = CLAHE_CLIP,
    tile_grid_size: tuple = CLAHE_TILE,
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalisation (CLAHE).

    Uses scikit-image's equalize_adapthist which expects a float image in [0, 1].
    The output is rescaled back to uint8 [0, 255].

    Args:
        img_gray       : Normalised uint8 grayscale image.
        clip_limit     : CLAHE clip limit (default CLAHE_CLIP = 2.0).
        tile_grid_size : Tile size for local histogram (default (8, 8)).

    Returns:
        CLAHE-enhanced uint8 image.
    """
    img_norm  = img_gray.astype(np.float32) / 255.0
    img_clahe = exposure.equalize_adapthist(
        img_norm,
        clip_limit=clip_limit / 100.0,
        kernel_size=tile_grid_size,
    )
    return (img_clahe * 255).astype(np.uint8)


def is_dull(img_gray: np.ndarray, threshold: int = DULL_THRESHOLD) -> bool:
    """
    Return True if the image is underexposed (mean pixel intensity < threshold).

    Only dull images receive CLAHE so well-exposed X-rays are not over-processed.

    Args:
        img_gray  : Grayscale image array.
        threshold : Mean-pixel threshold below which CLAHE is applied.

    Returns:
        bool – True if the image is considered underexposed.
    """
    return img_gray.mean() < threshold


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def build_or_load_enhanced_df(df: pd.DataFrame, force_enhance: bool = False) -> pd.DataFrame:
    """
    Build (first run) or reload (subsequent runs) the enhanced image dataset.

    First-run behaviour  (USING_KAGGLE_DATASET = False):
      - Iterates over raw image paths in df.
      - Applies percentile normalisation to every image.
      - Applies CLAHE only to dull images (or all if force_enhance=True).
      - Saves processed images to LOCAL_ENHANCED_DIR.
      - Saves a metadata CSV to CSV_SAVE_PATH.

    Reuse behaviour (USING_KAGGLE_DATASET = True):
      - Reads the pre-built metadata CSV from Kaggle input.
      - Verifies all image paths exist; warns if any are missing.

    Args:
        df             : Base dataframe with columns [ImageID, BoneAge, Gender, ImagePath].
        force_enhance  : If True, apply CLAHE to every image regardless of dullness.

    Returns:
        DataFrame extended with [EnhancedPath, WasEnhanced, MeanBefore, MeanAfter].
    """
    if USING_KAGGLE_DATASET:
        print(f"Loading metadata from Kaggle dataset: {CSV_PATH}")
        df_loaded = pd.read_csv(CSV_PATH)
        missing = df_loaded["EnhancedPath"].apply(lambda p: not os.path.exists(p)).sum()
        if missing:
            print(f"⚠  Warning: {missing} enhanced image paths not found. Check KAGGLE_ENHANCED_DIR.")
        else:
            print(f"✓  All {len(df_loaded)} enhanced images found in {KAGGLE_ENHANCED_DIR}")
        return df_loaded

    print("Processing and enhancing images → saving to /kaggle/working (first-time run)…")
    os.makedirs(LOCAL_ENHANCED_DIR, exist_ok=True)

    enhanced_paths, was_enhanced, mean_before, mean_after = [], [], [], []

    for img_path in tqdm(df["ImagePath"], desc="Enhancing X-Rays"):
        fname    = os.path.basename(img_path)
        out_path = os.path.join(LOCAL_ENHANCED_DIR, fname)

        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            enhanced_paths.append(None); was_enhanced.append(None)
            mean_before.append(None);    mean_after.append(None)
            continue

        raw_mean = float(img_gray.mean())
        img_norm = percentile_normalize(img_gray)
        dull     = is_dull(img_norm)

        if not os.path.exists(out_path):
            img_out = apply_clahe(img_norm) if (force_enhance or dull) else img_norm
            cv2.imwrite(out_path, img_out)
        else:
            img_out = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)

        kaggle_path = os.path.join(KAGGLE_ENHANCED_DIR, fname)
        enhanced_paths.append(kaggle_path)
        was_enhanced.append(bool(force_enhance or dull))
        mean_before.append(round(raw_mean, 2))
        mean_after.append(round(float(img_out.mean()), 2))

    df = df.copy()
    df["EnhancedPath"] = enhanced_paths
    df["WasEnhanced"]  = was_enhanced
    df["MeanBefore"]   = mean_before
    df["MeanAfter"]    = mean_after
    df.to_csv(CSV_SAVE_PATH, index=False)

    n_enhanced = df["WasEnhanced"].sum()
    print(f"✓  {n_enhanced}/{len(df)} images CLAHE-enhanced.")
    print(f"   Images saved → {LOCAL_ENHANCED_DIR}")
    print(f"   CSV saved    → {CSV_SAVE_PATH}")
    print("\nNext steps:")
    print("  1. Download the folder and CSV from /kaggle/working")
    print("  2. Upload both as a new Kaggle dataset")
    print(f"  3. Set USING_KAGGLE_DATASET = True and update KAGGLE_ENHANCED_DIR")
    return df