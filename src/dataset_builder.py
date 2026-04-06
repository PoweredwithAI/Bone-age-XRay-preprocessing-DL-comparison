"""
dataset_builder.py
------------------
Loads and standardises the RSNA Bone Age CSV into a clean DataFrame.

Adds a typed ImagePath column so all downstream pipeline steps receive a single
consistent dataframe schema: [ImageID, BoneAge, Gender, ImagePath].
"""

import os
import pandas as pd

# Default raw dataset path on Kaggle
RSNA_CSV = (
    "/kaggle/input/datasets/kmader/rsna-bone-age"
    "/boneage-training-dataset.csv"
)
RSNA_IMG_DIR = (
    "/kaggle/input/datasets/kmader/rsna-bone-age"
    "/boneage-training-dataset/boneage-training-dataset"
)


def load_rsna_dataframe(
    csv_path: str = RSNA_CSV,
    img_dir:  str = RSNA_IMG_DIR,
) -> pd.DataFrame:
    """
    Load the raw RSNA Bone Age CSV and return a clean, typed DataFrame.

    Standardises column names (id → ImageID, boneage → BoneAge, male → Gender),
    converts the boolean Gender column to int (1 = Male, 0 = Female), and
    appends a full ImagePath for every record.

    Args:
        csv_path : Path to boneage-training-dataset.csv.
        img_dir  : Directory containing the raw .png X-ray images.

    Returns:
        pd.DataFrame with columns:
          ImageID (int64), BoneAge (int64), Gender (int64), ImagePath (str).
    """
    print("=" * 60)
    print("LOADING RSNA BONE AGE DATASET")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    df = df.rename(columns={"id": "ImageID", "boneage": "BoneAge", "male": "Gender"})
    df["Gender"]    = df["Gender"].astype(int)   # True → 1, False → 0
    df["ImagePath"] = df["ImageID"].apply(lambda x: os.path.join(img_dir, f"{x}.png"))

    print(df.head())
    print("-" * 60)
    print(f"Total samples  : {len(df)}")
    print(f"Gender dist    :\n{df['Gender'].value_counts().rename({1: 'Male', 0: 'Female'})}")
    print(f"Age (months)   :\n{df['BoneAge'].describe()}")
    print("=" * 60)
    return df