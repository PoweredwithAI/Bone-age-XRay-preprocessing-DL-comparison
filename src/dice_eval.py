# src/dice_eval.py
"""
DICE Score Evaluation — FSCNN Predicted Masks vs Manual Ground-Truth Masks
==========================================================================
Evaluates segmentation quality of the trained FSCNN checkpoint by comparing
predicted hand masks against Zenodo manual ground-truth masks.

Column names match notebook-2 exactly:
    df columns used : ImageID, EnhancedPath
    Model variable  : modelmodule  (loaded FSCNNLightning, eval mode)
    Predict function: fscnn_predict_mask(image_path, input_size, thresh)

Position in pipeline
---------------------
  Stage 1  → preprocessing.py   (CLAHE enhancement)
  Stage 2a → fscnn_train.py     (FSCNN training)
  Stage 2b → fscnn_infer.py     (artefact removal)
  ► HERE   → dice_eval.py       (mask quality validation)
  Stage 3  → (future) model training on raw / enhanced / cleaned datasets

Key design decisions
---------------------
- eval_df is pre-filtered to rows where a manual mask file exists using a
  single os.listdir() scan instead of per-row os.path.exists() calls.
  Reduces filesystem calls from N_total → 1 and loop iterations from
  12,611 → N_masked (approx 9,000 for this dataset).
- DICE handles resolution mismatches via nearest-neighbour resize before
  comparison, so predicted and ground-truth masks need not be the same size.
- Laplace smoothing (1e-6) prevents division-by-zero on degenerate masks.
- Results are left-merged back onto df so downstream training scripts can
  optionally filter on PredictedDICE quality.

References
----------
- Rassmann et al. 2024 : https://doi.org/10.1007/s00247-023-05789-1
- Zenodo masks         : https://zenodo.org/records/7611677
- FSCNN repo           : https://github.com/aimi-bonn/hand-segmentation
"""

import os

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


# ── Config loader ──────────────────────────────────────────────────────────────

def _load_config(config_path: str = "configs/preprocessing.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── DICE function ──────────────────────────────────────────────────────────────

def dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    binarise_at: int = 127,
    smooth: float = 1e-6,
) -> float:
    """
    Compute DICE similarity coefficient between two binary masks.

    DICE = (2 * |pred ∩ gt|) / (|pred| + |gt|)

    Parameters
    ----------
    pred_mask   : Predicted mask (H, W) — uint8 0/255 or float probability.
    gt_mask     : Manual ground-truth mask (H, W) — uint8 0/255.
    binarise_at : Pixel threshold for binarisation (default 127).
    smooth      : Laplace smoothing term (default 1e-6).

    Returns
    -------
    float in [0, 1].  1.0 = perfect overlap.
    """
    pred_bin = (pred_mask > binarise_at).astype(np.uint8)
    gt_bin   = (gt_mask   > binarise_at).astype(np.uint8)

    # Resize gt to pred resolution if they differ
    if pred_bin.shape != gt_bin.shape:
        gt_bin = cv2.resize(
            gt_bin,
            (pred_bin.shape[1], pred_bin.shape[0]),   # cv2 wants (W, H)
            interpolation=cv2.INTER_NEAREST,
        )

    intersection = (pred_bin & gt_bin).sum()
    return float(
        (2.0 * intersection + smooth) /
        (pred_bin.sum() + gt_bin.sum() + smooth)
    )


# ── Mask predictor ─────────────────────────────────────────────────────────────
# Mirrors fscnn_predict_mask() from notebook Cell 14 exactly.
# Defined here so this module is self-contained when run as a standalone script.
# In the notebook, fscnn_predict_mask() from the cell above is used directly.

def predict_mask(
    image_path: str,
    model,
    device: torch.device,
    input_size: int = 512,
    threshold: float = 0.5,
) -> tuple:
    """
    Run FSCNN on one enhanced X-ray and return a full-resolution binary mask.

    Parameters
    ----------
    image_path : Path to the enhanced greyscale PNG.
    model      : FSCNNLightning instance in eval mode.
    device     : torch.device.
    input_size : Model input resolution — must match training size (512).
    threshold  : Foreground probability threshold (default 0.5).

    Returns
    -------
    (mask, img_u8) at original resolution, or (None, None) on failure.
    """
    if not os.path.exists(image_path):
        return None, None

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None, None

    h0, w0 = img_gray.shape
    img_u8  = img_gray.astype(np.uint8)

    pil_resized = Image.fromarray(img_u8).resize(
        (input_size, input_size), Image.BILINEAR
    )
    arr = np.array(pil_resized, dtype=np.float32) / 255.0
    x   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, dict) and "logits" in logits:
            logits = logits["logits"]
        prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

    mask_small = (prob >= threshold).astype(np.uint8) * 255
    mask       = cv2.resize(
        mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST
    )
    return mask, img_u8


# ── Core evaluation function ───────────────────────────────────────────────────

def run_dice_evaluation(
    df: pd.DataFrame,
    model,
    device: torch.device,
    config_path: str = "configs/preprocessing.yaml",
    n_samples: int = None,
    notebook_mode: bool = False,
) -> tuple:
    """
    Evaluate FSCNN mask quality vs manual ground-truth masks.

    Pre-filters df to rows with an available manual mask using a single
    os.listdir() call — avoids iterating the full 12,611-row dataset when
    only a subset has ground-truth masks.

    Parameters
    ----------
    df            : DataFrame with columns ImageID and EnhancedPath.
    model         : FSCNNLightning in eval mode.
    device        : torch.device.
    config_path   : Path to YAML config file.
    n_samples     : Random sample size (None = all masked rows).
    notebook_mode : True enables tqdm.notebook progress bar for Kaggle/Jupyter.

    Returns
    -------
    dice_df : Per-image results — ImageID, FileName, PredictedDICE,
              HasManualMask.
    df      : Input df with PredictedDICE and HasManualMask left-merged in.
              Rows without masks receive NaN.
    """
    cfg        = _load_config(config_path)
    mask_dir   = cfg["kaggle_paths"]["mask_dir"]
    input_size = cfg["fscnn_inference"]["image_size"]
    threshold  = cfg["fscnn_inference"]["threshold"]

    # ── Single listdir scan — build set of masked ImageIDs ───────────────────
    mask_ids = {
        os.path.splitext(f)[0]              # "1377.png" → "1377"
        for f in os.listdir(mask_dir)
        if f.lower().endswith(".png")
    }

    eval_df = (
        df[df["ImageID"].astype(str).isin(mask_ids)]
        .reset_index(drop=True)
    )

    print(f"[dice_eval] Mask directory  : {len(mask_ids):,} mask files found")
    print(f"[dice_eval] df rows matched : {len(eval_df):,}  "
          f"(skipping {len(df) - len(eval_df):,} rows without a mask)")

    if n_samples:
        eval_df = eval_df.sample(n_samples, random_state=42).reset_index(drop=True)
        print(f"[dice_eval] Sampling        : {n_samples} rows selected")

    # ── Progress bar ──────────────────────────────────────────────────────────
    _tqdm = tqdm
    if notebook_mode:
        try:
            from tqdm.notebook import tqdm as tqdm_nb
            _tqdm = tqdm_nb
        except ImportError:
            pass

    # ── Evaluation loop ───────────────────────────────────────────────────────
    # Every row in eval_df is guaranteed to have a mask — no per-row
    # os.path.exists() check needed inside the loop.
    results = []

    for _, row in _tqdm(
        eval_df.iterrows(), total=len(eval_df), desc="DICE Evaluation"
    ):
        img_path = row["EnhancedPath"]
        fname    = os.path.basename(img_path)
        gt_path  = os.path.join(mask_dir, fname)

        pred_mask, _ = predict_mask(
            img_path, model, device, input_size, threshold
        )

        if pred_mask is not None:
            gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            score   = dice_score(pred_mask, gt_mask)
        else:
            score = None

        results.append({
            "ImageID"      : row["ImageID"],
            "FileName"     : fname,
            "PredictedDICE": round(score, 4) if score is not None else None,
            "HasManualMask": True,  # guaranteed by eval_df construction
        })

    dice_df = pd.DataFrame(results)
    scored  = dice_df.dropna(subset=["PredictedDICE"])

    # ── Summary ───────────────────────────────────────────────────────────────
    _print_summary(dice_df, scored)

    # ── Merge back onto df (left join) ────────────────────────────────────────
    df = df.merge(
        dice_df[["ImageID", "PredictedDICE", "HasManualMask"]],
        on="ImageID",
        how="left",
    )
    print(f"\n[dice_eval] Scores merged onto df — "
          f"{df['PredictedDICE'].notna().sum():,} rows have a DICE score.")

    return dice_df, df


# ── Summary printer ────────────────────────────────────────────────────────────

def _print_summary(dice_df: pd.DataFrame, scored: pd.DataFrame) -> None:
    print("=" * 52)
    print("        DICE Score Evaluation Summary")
    print("=" * 52)
    print(f"  Images evaluated    : {len(dice_df):,}")
    print(f"  Successfully scored : {len(scored):,}")
    print(f"  Failed (read error) : {dice_df['PredictedDICE'].isna().sum():,}")
    print("-" * 52)
    print(f"  Mean   DICE         : {scored['PredictedDICE'].mean():.4f}")
    print(f"  Median DICE         : {scored['PredictedDICE'].median():.4f}")
    print(f"  Std    DICE         : {scored['PredictedDICE'].std():.4f}")
    print(f"  Min    DICE         : {scored['PredictedDICE'].min():.4f}")
    print(f"  Max    DICE         : {scored['PredictedDICE'].max():.4f}")
    print("-" * 52)
    print(f"  DICE ≥ 0.99  (%)    : "
          f"{(scored['PredictedDICE'] >= 0.99).mean()*100:.1f}%")
    print(f"  DICE ≥ 0.95  (%)    : "
          f"{(scored['PredictedDICE'] >= 0.95).mean()*100:.1f}%")
    print(f"  DICE ≥ 0.90  (%)    : "
          f"{(scored['PredictedDICE'] >= 0.90).mean()*100:.1f}%")
    print(f"  DICE  < 0.90 (%)    : "
          f"{(scored['PredictedDICE'] <  0.90).mean()*100:.1f}%")
    print("=" * 52)


# ── Save results ───────────────────────────────────────────────────────────────

def save_dice_results(
    df: pd.DataFrame,
    dice_df: pd.DataFrame,
    output_dir: str = "/kaggle/working",
) -> None:
    """
    Save both the merged df and the standalone dice_df to CSV.

    Parameters
    ----------
    df         : Full dataframe with PredictedDICE merged in.
    dice_df    : Per-image DICE scores only.
    output_dir : Directory to write CSVs (default /kaggle/working).
    """
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, "RSNAboneage_with_dice.csv")
    dice_path = os.path.join(output_dir, "RSNAboneage_dice_scores_only.csv")

    df.to_csv(full_path, index=False)
    dice_df.to_csv(dice_path, index=False)

    print(f"[dice_eval] Full df saved   → {full_path}  | shape: {df.shape}")
    print(f"[dice_eval] DICE scores     → {dice_path}  | shape: {dice_df.shape}")


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_dice_distribution(
    dice_df: pd.DataFrame,
    save_path: str = None,
) -> None:
    """
    Two-panel plot: DICE histogram + sorted per-image score curve.

    Parameters
    ----------
    dice_df   : Output of run_dice_evaluation().
    save_path : If provided, saves figure to this path instead of showing.
    """
    scored = dice_df.dropna(subset=