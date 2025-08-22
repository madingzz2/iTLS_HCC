#D:/conda  python3
"""
extract_radiomics.py
"""
from __future__ import annotations
import os, json, argparse, itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import SimpleITK as sitk
import pandas as pd
from radiomics import featureextractor, setVerbosity
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=Path, default="dataset")
    parser.add_argument("-o", "--out_csv", type=Path, default="all_features_600.csv")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count())
    args = parser.parse_args()

    tasks = build_tasks(args.dataset)
    print(f"Total {len(tasks)} samples (≈ 600×3={1800})")

    records = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(extract_one, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                records.append(fut.result())
            except Exception as e:
                print("ERROR:", e)

    df = pd.DataFrame.from_records(records)
    cols = ["PatientID"] + [c for c in df.columns if c != "PatientID"]
    df = df[cols]
    df.to_csv(args.out_csv, index=False)
    print("Saved ->", args.out_csv.resolve())

setVerbosity(60)

PARAMS = {
    # 基础
    "binWidth": 25,
    "resampledPixelSpacing": [1, 1, 1],
    "interpolator": "sitkBSpline",
    "normalize": True,
    "normalizeScale": 100,
    "removeOutliers": 1,
    "minimumROISize": 10,

    "enableCExtensions": True,
    "enableAllFeatures": True,
    "enableAllImageTypes": True,

    "imageType": {
        "Original": {},
        "LoG": {"sigma": [1.0, 2.0, 3.0, 4.0, 5.0]},
        "Wavelet": {},
        "Square": {},
        "SquareRoot": {},
        "Logarithm": {},
        "Exponential": {},
        "Gradient": {},
        "LBP2D": {},
        "LBP3D": {}
    }
}

MODALITIES = ["AP", "PVP", "T2WI"]


def sitk_read(p: Path) -> sitk.Image:
    p = Path(p)
    if p.suffix.lower() in [".nii", ".gz"]:
        return sitk.ReadImage(str(p))
    # DICOM 目录
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(p)))
    return reader.Execute()

def extract_one(c: dict) -> dict:
    extractor = featureextractor.RadiomicsFeatureExtractor(**c["params"])
    image = sitk_read(c["image"])
    mask = sitk_read(c["mask"])
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    feats = extractor.execute(image, mask)

    modality = c["modality"]
    row = {}
    row["PatientID"] = c["pid"]
    for k, v in feats.items():
        if k.startswith("original_") or k.startswith("wavelet") or \
           k.startswith("log-sigma") or k.startswith("square") or \
           k.startswith("squareroot") or k.startswith("logarithm") or \
           k.startswith("exponential") or k.startswith("gradient") or \
           k.startswith("lbp"):
            new_key = f"{k}_{modality}"
            row[new_key] = v
    return row

def build_tasks(dataset_dir: Path) -> list[dict]:
    tasks = []
    for mod in MODALITIES:
        img_dir = dataset_dir / mod / "images"
        msk_dir = dataset_dir / mod / "masks"
        for img_p in sorted(img_dir.glob("*")):
            msk_p = msk_dir / img_p.name
            if not msk_p.exists():
                continue
            pid = img_p.stem.split(".")[0]
            tasks.append({
                "pid": pid,
                "modality": mod,
                "image": img_p,
                "mask": msk_p,
                "params": PARAMS
            })
    return tasks



if __name__ == "__main__":
    main()