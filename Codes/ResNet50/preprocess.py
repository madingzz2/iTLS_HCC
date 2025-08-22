
import os, argparse, cv2
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from skimage.transform import resize

def extract_tumor_3slice(img_path, mask_path, out_path, size=224):
    img  = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))   # (D,H,W)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    mask = (mask > 0).astype(np.uint8)

    z_idx = np.where(mask.sum(axis=(1, 2)) > 0)[0]
    if len(z_idx) == 0:
        return  # skip without lesion
    mid   = int(np.median(z_idx))
    idx   = [max(mid-2, 0), mid, min(mid+2, img.shape[0]-1)]
    stack = img[idx]                                # (3,H,W)
    stack = resize(stack, (3, size, size), preserve_range=True, anti_aliasing=True)
    np.save(out_path, stack.astype(np.float32))

def run(root, out, modals):
    root = Path(root)
    out  = Path(out)
    for mod in modals:
        img_dir  = root / mod / 'images'
        mask_dir = root / mod / 'masks'
        dst_dir  = out  / mod
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_p in img_dir.glob('*.nii.gz'):
            mask_p = mask_dir / img_p.name
            if not mask_p.exists():
                continue
            out_p = dst_dir / (img_p.stem + '.npy')
            extract_tumor_3slice(img_p, mask_p, out_p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True,
                        help='path/to/input')
    parser.add_argument('--out',  required=True,
                        help='path/to/output')
    parser.add_argument('--modal', nargs='+', required=True,
                        help='Model names')
    args = parser.parse_args()
    run(args.root, args.out, args.modal)