import os, argparse, torch, numpy as np, pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from torchvision import transforms
from model import ResNet50_3slice

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, modality, transform):
        self.root = Path(root) / modality
        self.files = sorted(self.root.glob('*.npy'))
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = np.load(self.files[idx])
        label = int(self.files[idx].stem.split('_')[-1])
        img = torch.from_numpy(img.astype(np.float32))
        img = self.transform(img)
        return img, label, self.files[idx].stem

def extract(modality, root, model_root, out_dir, dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    dataset = Dataset(root, modality, transform)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model = ResNet50_3slice(num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(Path(model_root)/modality/'best.pth', map_location=device))
    model.eval()

    feats, labels, ids = [], [], []
    with torch.no_grad():
        for imgs, lbs, names in loader:
            imgs = imgs.to(device)
            f = model.forward_features(imgs).cpu().numpy()
            feats.append(f)
            labels.extend(lbs)
            ids.extend(names)
    feats = np.vstack(feats)
    df = pd.DataFrame(feats, columns=[f'F{i+1}_{modality}' for i in range(2048)])
    df.insert(0, 'ID', ids)
    df['label'] = labels

    # 压缩 1500
    pca = PCA(n_components=dim, random_state=42)
    reduced = pca.fit_transform(df.drop(columns=['ID', 'label']))
    cols = [f'F{i+1}_{modality}' for i in range(dim)]
    out = pd.DataFrame(reduced, columns=cols)
    out.insert(0, 'ID', ids)
    out['label'] = labels

    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir) / f'{modality}_DL_features.csv'
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'{modality} 1500-d saved -> {out_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--model_root', default='models')
    parser.add_argument('--out', default='features')
    parser.add_argument('--dim', type=int, default=1500)
    args = parser.parse_args()
    extract(args.modal, args.root, args.model_root, args.out, args.dim)