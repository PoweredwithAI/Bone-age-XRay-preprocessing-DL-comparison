"""
visualize.py
------------
Utility functions for exploratory and diagnostic visualisation.

  - sample_raw_grid          : Grid of random raw X-rays
  - compare_enhancement      : Side-by-side raw vs. enhanced pairs
  - compare_all_three        : Three-column raw / enhanced / cleaned comparison
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def sample_raw_grid(img_dir: str, n: int = 100, ncols: int = 10, figsize=(15, 15)) -> None:
    """
    Display a grid of n randomly sampled raw X-rays from img_dir.

    Args:
        img_dir : Directory of raw .png images.
        n       : Number of images to sample (default 100).
        ncols   : Number of columns in the grid (default 10).
        figsize : Matplotlib figure size.
    """
    files  = random.sample(os.listdir(img_dir), n)
    nrows  = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    for i, fname in enumerate(files):
        ax  = axes[i // ncols, i % ncols]
        img = Image.open(os.path.join(img_dir, fname)).convert("L")
        ax.imshow(img, cmap="gray")
        ax.set_title(fname[:8], fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def compare_enhancement(
    df: pd.DataFrame,
    n: int = 10,
    save_path: str = None,
) -> None:
    """
    Side-by-side comparison of raw vs. CLAHE-enhanced X-rays.

    Args:
        df        : DataFrame with ImagePath and EnhancedPath columns.
        n         : Number of pairs to display (default 10).
        save_path : If given, saves the figure to this path.
    """
    sample = df.sample(n, random_state=42).reset_index(drop=True)
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

    for i, row in sample.iterrows():
        orig = cv2.imread(row["ImagePath"],    cv2.IMREAD_GRAYSCALE)
        enh  = cv2.imread(row["EnhancedPath"], cv2.IMREAD_GRAYSCALE)
        axes[i, 0].imshow(orig, cmap="gray"); axes[i, 0].set_title(f"Raw  ID={row['ImageID']}")
        axes[i, 1].imshow(enh,  cmap="gray"); axes[i, 1].set_title("Enhanced (CLAHE)")
        for ax in axes[i]: ax.axis("off")

    plt.suptitle("Raw vs. Enhanced X-rays", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def compare_all_three(
    raw_dir:   str,
    prep_dir:  str,
    clean_dir: str,
    n: int = 10,
    save_path: str = None,
) -> None:
    """
    Three-column comparison: Raw | Enhanced | Cleaned (artifacts removed).

    Args:
        raw_dir   : Directory of original raw X-rays.
        prep_dir  : Directory of CLAHE-enhanced X-rays.
        clean_dir : Directory of FSCNN-cleaned (delabeled) X-rays.
        n         : Number of rows to display (default 10).
        save_path : If given, saves the figure.
    """
    files  = [f for f in os.listdir(prep_dir) if f.lower().endswith(".png")]
    sample = random.sample(files, min(n, len(files)))
    cols   = ["Raw X-ray", "Enhanced (CLAHE)", "Cleaned (Artifacts Removed)"]

    fig, axes = plt.subplots(len(sample), 3, figsize=(12, 4 * len(sample)))
    for ax, label in zip(axes[0], cols):
        ax.set_title(label, fontsize=12, fontweight="bold")

    for i, fname in enumerate(sample):
        for j, d in enumerate([raw_dir, prep_dir, clean_dir]):
            img = cv2.imread(os.path.join(d, fname), cv2.IMREAD_GRAYSCALE)
            axes[i, j].imshow(img if img is not None else np.zeros((64, 64), dtype=np.uint8),
                              cmap="gray")
            axes[i, j].axis("off")
        axes[i, 0].set_ylabel(fname[:12], fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()