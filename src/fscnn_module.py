"""
fscnn_module.py
---------------
Step 2: FSCNN (FastSurferCNN) hand-segmentation module.

Wraps the FastSurferCNN U-Net architecture from the aimi-bonn/hand-segmentation
repo in a PyTorch Lightning module for training and inference.

Reference:
  Rassmann S, et al. "Deeplasia: deep learning for bone age assessment validated
  on skeletal dysplasias." Pediatr Radiol 54, 82–95 (2024).
  https://doi.org/10.1007/s00247-023-05789-1

  Masks sourced from: https://zenodo.org/records/7611677
  FSCNN repo:         https://github.com/aimi-bonn/hand-segmentation/tree/master/FSCNN/lib
"""

import os
import sys
import random
import shutil

import torch
import torch.nn as nn
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw

# ─── FSCNN repo must be cloned and on sys.path before importing below ─────────
# Run once:  !git clone https://github.com/aimi-bonn/hand-segmentation.git
# Then:      sys.path.append("/kaggle/working/hand-segmentation/FSCNN")
# from lib.datasets import MaskModule, MaskDataSet
# from lib.models import FastSurferCNN, CombinedLoss


# ─── Augmentation patches ─────────────────────────────────────────────────────

def patched_get_default_train_aug(size: int = 512) -> A.Compose:
    """
    Rich augmentation pipeline for FSCNN training.

    Replaces the default augmentation in MaskModule with a more aggressive set
    that improves generalisation on RSNA hand X-rays:
      - Vertical / horizontal flips (p=0.5)
      - Random 90° rotations
      - Random brightness / contrast jitter (±30 %)
      - Random resized crop (scale 80–100 %, aspect 0.8–1.25)
      - Gaussian noise (std 0.04–0.16)
      - JPEG compression simulation (quality 25–80)

    Args:
        size : Target spatial resolution (H = W). Default 512.

    Returns:
        Albumentations Compose pipeline.
    """
    size_tuple = (size, size)
    return A.Compose(
        [
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                p=0.5,
                brightness_limit=0.3,
                contrast_limit=0.3,
                brightness_by_max=True,
            ),
            A.RandomResizedCrop(
                size=size_tuple,
                scale=(0.8, 1.0),
                ratio=(0.8, 1.25),
            ),
            A.GaussNoise(std_range=(0.04, 0.16), p=0.5),
            A.ImageCompression(quality_range=(25, 80), p=0.5),
            ToTensorV2(),
        ],
        additional_targets={"weight": "mask"},
    )


def patched_get_inference_aug(size: int = 512) -> A.Compose:
    """
    Minimal augmentation for FSCNN inference (resize + centre-crop only).

    Args:
        size : Target spatial resolution. Default 512.

    Returns:
        Albumentations Compose pipeline.
    """
    size_tuple = (size, size)
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=size_tuple,
                scale=(1.0, 1.0),
                ratio=(1.0, 1.0),
            ),
            ToTensorV2(),
        ],
        additional_targets={"weight": "mask"},
    )


def patched_alex_aug(img: Image.Image) -> Image.Image:
    """
    Draw a random black rectangle on the image for occlusion augmentation.

    Patched version of MaskDataSet.alex_aug that ensures x1 > x0 and y1 > y0
    to avoid degenerate (zero-area) rectangles that caused errors in the
    original implementation.

    Args:
        img : PIL Image to augment.

    Returns:
        PIL Image with a random black rectangle drawn.
    """
    draw = ImageDraw.Draw(img)
    w, h = img.size

    x0, x1 = sorted([random.randint(0, w), random.randint(0, w)])
    y0, y1 = sorted([random.randint(0, h), random.randint(0, h)])

    if x0 == x1:
        x1 = min(x1 + 1, w)
    if y0 == y1:
        y1 = min(y1 + 1, h)

    draw.rectangle([x0, y0, x1, y1], fill=0)
    return img


# ─── Lightning Module ─────────────────────────────────────────────────────────

class FSCNNLightning(pl.LightningModule):
    """
    PyTorch Lightning wrapper around FastSurferCNN for hand segmentation.

    Architecture:
      - FastSurferCNN (U-Net-like) with 1 input channel (grayscale) and 2
        output classes (background / hand).
      - Loss: CombinedLoss = 1.0 × Dice + 1.0 × Cross-Entropy.
      - Optimiser: Adam with ReduceLROnPlateau (factor=0.5, patience=3 epochs).

    Args:
        n_classes : Number of segmentation classes (default 2: background + hand).
        lr        : Initial learning rate (default 1e-3).
    """

    def __init__(self, n_classes: int = 2, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Import here so the module can be imported without the FSCNN repo on path
        from lib.models import FastSurferCNN, CombinedLoss  # noqa: F401

        self.net = FastSurferCNN(
            num_classes=n_classes,
            num_input_channels=1,
            kernel_size=(5, 5),
            num_filters=64,
        )
        self.criterion = CombinedLoss(weight_dice=1.0, weight_ce=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["image"]
        y = batch["mask"].long()
        w = batch["weight"]
        logits = self(x)
        loss, dice_val, ce_val = self.criterion(logits, y, w)
        self.log("train_loss", loss, on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train_dice", dice_val, on_step=False, on_epoch=True)
        self.log("train_ce",   ce_val,   on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x = batch["image"]
        y = batch["mask"].long()
        w = batch["weight"]
        logits = self(x)
        loss, dice_val, ce_val = self.criterion(logits, y, w)
        self.log("val_loss", loss,     on_epoch=True, prog_bar=True)
        self.log("val_dice", dice_val, on_epoch=True, prog_bar=True)
        self.log("val_ce",   ce_val,   on_epoch=True)
        return loss

    def configure_optimizers(self) -> dict:
        """Adam + ReduceLROnPlateau (halves LR after 3 stagnant val_loss epochs)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


# ─── RGBA training-image builder ──────────────────────────────────────────────

def build_rgba_training_images(
    rsna_img_dir: str,
    mask_dir: str,
    rgba_out_dir: str,
) -> int:
    """
    Pair enhanced X-rays with their manual masks into RGBA tensors for FSCNN training.

    The RGBA format packs (R=G=B=grayscale_image, A=binary_mask) so that
    MaskDataSet can load image and mask from a single PNG file.

    Args:
        rsna_img_dir  : Directory of enhanced (CLAHE-normalised) X-ray PNGs.
        mask_dir      : Directory of manual binary mask PNGs (from Zenodo).
        rgba_out_dir  : Output directory for RGBA PNGs.

    Returns:
        Number of matched RGBA images written.
    """
    import cv2, numpy as np
    from PIL import Image
    from tqdm import tqdm

    os.makedirs(rgba_out_dir, exist_ok=True)
    rsna_files = [f for f in os.listdir(rsna_img_dir) if f.lower().endswith(".png")]
    matched = 0

    for fname in tqdm(rsna_files, desc="Building RGBA training images"):
        img_path  = os.path.join(rsna_img_dir, fname)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            continue

        img  = cv2.imread(img_path,  cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        if img.shape != mask.shape:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        img_u8   = img.astype(np.uint8)
        _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        rgb  = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
        rgba = np.dstack([rgb, mask_bin])
        Image.fromarray(rgba).save(os.path.join(rgba_out_dir, fname))
        matched += 1

    print(f"✓  Created {matched} matched RGBA training images → {rgba_out_dir}")
    return matched


def split_rgba_train_val(src_dir: str, base_out: str, val_fraction: float = 0.1) -> None:
    """
    Randomly split RGBA images into train/ and val/ sub-directories.

    Args:
        src_dir      : Directory containing RGBA PNGs.
        base_out     : Parent output directory; train/ and val/ created inside.
        val_fraction : Fraction of images held out for validation (default 0.10).
    """
    train_out = os.path.join(base_out, "train")
    val_out   = os.path.join(base_out, "val")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(val_out,   exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    random.shuffle(files)
    n_val    = int(val_fraction * len(files))
    val_set  = set(files[:n_val])

    for f in files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(val_out if f in val_set else train_out, f)
        shutil.copy2(src, dst)

    print(f"Split: {len(files) - n_val} train / {n_val} val → {base_out}")