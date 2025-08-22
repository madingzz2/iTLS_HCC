import os, argparse, json
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from pathlib import Path
from model import ResNet50_3slice

class SliceDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        self.files, self.labels = files, labels
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img = np.load(self.files[idx])
        img = torch.from_numpy(img.astype(np.float32))
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

def train_fold(modality, root, seed, epochs, batch_size, lr):
    root = Path(root) / modality
    files = list(root.glob('*.npy'))
    patients = [f.stem for f in files]
    labels = [int(p.split('_')[-1]) for p in patients]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(patients, labels)):
        print(f'=== {modality}  Fold {fold+1}/10 ===')
        train_files = [files[i] for i in train_idx]
        val_files   = [files[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels   = [labels[i] for i in val_idx]

        transform = transforms.Normalize(mean=[0.485,0.456,0.406],
                                         std =[0.229,0.224,0.225])
        train_ds = SliceDataset(train_files, train_labels, transform)
        val_ds   = SliceDataset(val_files,   val_labels,   transform)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet50_3slice(num_classes=2, pretrained=True).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0
        save_dir = Path('models') / modality
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, epochs+1):
            model.train()
            preds_all, labels_all = [], []
            for imgs, lbs in train_dl:
                imgs, lbs = imgs.to(device), lbs.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = criterion(logits, lbs)
                loss.backward()
                optimizer.step()
                preds_all.extend(logits.argmax(1).cpu().numpy())
                labels_all.extend(lbs.cpu().numpy())
            train_acc = accuracy_score(labels_all, preds_all)

            # val
            model.eval()
            preds_all, labels_all = [], []
            with torch.no_grad():
                for imgs, lbs in val_dl:
                    logits = model(imgs.to(device))
                    preds_all.extend(logits.argmax(1).cpu().numpy())
                    labels_all.extend(lbs.cpu().numpy())
            val_acc = accuracy_score(labels_all, preds_all)

            scheduler.step()
            torch.save(model.state_dict(), save_dir / 'last.pth')
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), save_dir / 'best.pth')
                print(f'{modality} fold{fold} epoch{epoch} best={best_acc:.4f}')
        torch.save(model.state_dict(), save_dir / f'best_fold{fold}.pth')
    print(f'{modality} 10-fold CV done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modal', required=True,
                        help='Modal name')
    parser.add_argument('--root', required=True,
                        help='path/to/input')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    train_fold(args.modal, args.root, args.seed, args.epochs, args.batch_size, args.lr)