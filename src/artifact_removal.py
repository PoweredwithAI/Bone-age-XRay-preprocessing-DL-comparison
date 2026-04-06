"""
artifact_removal.py
-------------------
Step 3: FSCNN-based artifact removal and image delabeling.

Uses a trained FSCNNLightning checkpoint to predict a binary hand mask for each
enhanced X-ray, then blanks (or inpaints) the non-hand background — removing
ruler labels, text overlays, and detector artefacts.

Two removal strategies are supported:
  - "blackout"  : Sets non-hand pixels to 0 (black). Fast, lossless for the ROI.
  - "inpaint"   : Fills non-hand region with TELEA inpainting. Smoother edges but slower.
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# ─── Kaggle dataset paths ─────────────────────────────────────────────────────
KAGGLE_CLEANED_DIR = (
    "/kaggle/input/datasets/ak7180979/delabeled-rsna-boneage-xrays"
    "/DelabeledRSNAboneageXRays/trainingdataset"
)
CSV_KAGGLE_PATH = (
    "/kaggle/input/datasets/ak7180979/cleaned-rsna-boneage-xrays"
    "/RSNAboneagecleanedtrainingdataset.csv"
)

LOCAL_CLEANED_DIR = "/kaggle/working/RSNAboneagecleanedtrainingdataset"
CSV_SAVE_PATH     = (
    "/kaggle/working/RSNAboneagecleaned"
    "/RSNAboneagecleanedtrainingdataset.csv"
)

USING_KAGGLE_DATASET = False
CLEANED_DIR = KAGGLE_CLEANED_DIR if USING_KAGGLE_DATASET else LOCAL_CLEANED_DIR
CSV_PATH    = CSV_KAGGLE_PATH    if USING_KAGGLE_DATASET else CSV_SAVE_PATH


# ─── Mask prediction ─────────────────────────────────────────────────────────

def fscnn_predict_mask(
    model_module,
    image_path: str,
    input_size: int = 512,
    thresh: float = 0.5,
    device: torch.device = None,
):
    """
    Run the FSCNN forward pass to predict a binary hand mask.

    The image is resized to (input_size × input_size) for the model, then the
    resulting mask is upsampled back to the original image resolution using
    nearest-neighbour interpolation.

    Args:
        model_module : Loaded FSCNNLightning instance (eval mode).
        image_path   : Path to the enhanced grayscale X-ray PNG.
        input_size   : Model input spatial resolution (default 512).
        thresh       : Foreground probability threshold (default 0.5).
        device       : torch.device; inferred from model if None.

    Returns:
        Tuple (mask_full_res, img_u8) or (None, None) on read failure.
          mask_full_res : Binary uint8 mask in original image resolution (0 or 255).
          img_u8        : Original grayscale image as uint8 array.
    """
    if device is None:
        device = next(model_module.parameters()).device

    if not os.path.exists(image_path):
        print(f"Missing file: {image_path}")
        return None, None

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Failed to read: {image_path}")
        return None, None

    h0, w0  = img_gray.shape
    img_u8  = img_gray.astype(np.uint8)
    pil_res = Image.fromarray(img_u8).resize((input_size, input_size), Image.BILINEAR)
    arr     = np.array(pil_res, dtype=np.float32) / 255.0
    x       = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model_module(x)
        if isinstance(logits, dict) and "logits" in logits:
            logits = logits["logits"]
        prob       = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
        mask_small = (prob > thresh).astype(np.uint8) * 255

    mask = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return mask, img_u8


# ─── Artifact removal ─────────────────────────────────────────────────────────

def remove_artifacts(
    img_u8: np.ndarray,
    mask: np.ndarray,
    strategy: str = "blackout",
) -> np.ndarray:
    """
    Remove non-hand background from an X-ray using a predicted binary mask.

    Args:
        img_u8   : Original grayscale image (uint8).
        mask     : Full-resolution binary mask (0 = background, 255 = hand).
        strategy : "blackout" zeros out non-hand pixels (default).
                   "inpaint"  fills them with TELEA inpainting for smoother edges.

    Returns:
        Cleaned grayscale uint8 image with background removed.

    Raises:
        ValueError if an unknown strategy is passed.
    """
    mask_bin = (mask > 127).astype(np.uint8)

    if strategy == "blackout":
        return cv2.bitwise_and(img_u8, img_u8, mask=mask_bin)

    if strategy == "inpaint":
        artifact_region = (1 - mask_bin).astype(np.uint8)
        return cv2.inpaint(img_u8, artifact_region, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    raise ValueError(f"Unknown strategy: '{strategy}'. Choose 'blackout' or 'inpaint'.")


def clean_xray(
    model_module,
    image_path: str,
    input_size: int = 512,
    thresh: float = 0.5,
    strategy: str = "blackout",
    device: torch.device = None,
):
    """
    End-to-end single-image cleaning pipeline: predict mask → remove background.

    Args:
        model_module : Loaded FSCNNLightning (eval mode).
        image_path   : Path to enhanced X-ray.
        input_size   : FSCNN input size.
        thresh       : Foreground threshold.
        strategy     : "blackout" or "inpaint".
        device       : Inference device.

    Returns:
        Tuple (img_u8, mask, clean) or (None, None, None) on failure.
    """
    mask, img_u8 = fscnn_predict_mask(model_module, image_path, input_size, thresh, device)
    if img_u8 is None or mask is None:
        return None, None, None
    clean = remove_artifacts(img_u8, mask, strategy)
    return img_u8, mask, clean


# ─── Orchestrator ─────────────────────────────────────────────────────────────

def build_or_load_cleaned_df(
    df: pd.DataFrame,
    model_module=None,
    input_size: int = 512,
    thresh: float = 0.5,
    strategy: str = "blackout",
    device: torch.device = None,
) -> pd.DataFrame:
    """
    Build (first run) or reload (subsequent runs) the delabeled image dataset.

    Mirrors the two-mode pattern of build_or_load_enhanced_df in preprocessing.py.
    Resume-safe: skips images whose output file already exists.

    Args:
        df           : DataFrame with EnhancedPath column from preprocessing step.
        model_module : FSCNNLightning in eval mode (required if first run).
        input_size   : FSCNN input resolution.
        thresh       : Foreground threshold for mask binarisation.
        strategy     : Background removal strategy ("blackout" | "inpaint").
        device       : Inference device.

    Returns:
        DataFrame extended with [CleanedPath, MeanBefore, MeanAfter, ProcessedOK].
    """
    if USING_KAGGLE_DATASET:
        print(f"Loading cleaned metadata from Kaggle dataset: {CSV_PATH}")
        df_loaded = pd.read_csv(CSV_PATH)
        missing = df_loaded["CleanedPath"].apply(lambda p: not os.path.exists(p)).sum()
        if missing:
            print(f"⚠  Warning: {missing} cleaned image paths not found.")
        else:
            print(f"✓  All {len(df_loaded)} cleaned images found in {KAGGLE_CLEANED_DIR}")
        return df_loaded

    print("Processing images with FSCNN → saving cleaned outputs to /kaggle/working…")
    os.makedirs(LOCAL_CLEANED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_SAVE_PATH), exist_ok=True)

    cleaned_paths, mean_before, mean_after, processed_ok = [], [], [], []

    for img_path in tqdm(df["EnhancedPath"], desc="Cleaning X-Rays"):
        fname    = os.path.basename(img_path)
        out_path = os.path.join(LOCAL_CLEANED_DIR, fname)

        if not os.path.exists(img_path):
            cleaned_paths.append(None); mean_before.append(None)
            mean_after.append(None);    processed_ok.append(False)
            continue

        if not os.path.exists(out_path):
            orig, mask, clean = clean_xray(model_module, img_path, input_size, thresh, strategy, device)
            if orig is None:
                cleaned_paths.append(None); mean_before.append(None)
                mean_after.append(None);    processed_ok.append(False)
                continue
            cv2.imwrite(out_path, clean)
            raw_mean   = float(orig.mean())
            clean_mean = float(clean.mean())
        else:
            orig  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
            clean = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
            if orig is None or clean is None:
                cleaned_paths.append(None); mean_before.append(None)
                mean_after.append(None);    processed_ok.append(False)
                continue
            raw_mean   = float(orig.mean())
            clean_mean = float(clean.mean())

        kaggle_path = os.path.join(KAGGLE_CLEANED_DIR, fname)
        cleaned_paths.append(kaggle_path)
        mean_before.append(round(raw_mean, 2))
        mean_after.append(round(clean_mean, 2))
        processed_ok.append(True)

    df = df.copy()
    df["CleanedPath"]  = cleaned_paths
    df["MeanBefore"]   = mean_before
    df["MeanAfter"]    = mean_after
    df["ProcessedOK"]  = processed_ok
    df.to_csv(CSV_SAVE_PATH, index=False)

    n_ok = int(df["ProcessedOK"].sum())
    print(f"✓  Successfully cleaned {n_ok}/{len(df)} images.")
    print(f"   Images saved → {LOCAL_CLEANED_DIR}")
    print(f"   CSV saved    → {CSV_SAVE_PATH}")
    print("\nNext steps:")
    print("  1. Download the folder and CSV from /kaggle/working")
    print("  2. Upload both as a new Kaggle dataset")
    print(f"  3. Set USING_KAGGLE_DATASET = True and update KAGGLE_CLEANED_DIR")
    return df