import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import models

from step10_dataset_raster import CapRasterDataset  # 直接复用你刚写的

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MAX = 7
TYPE_OUT_DIM = {0: 7, 1: 5, 2: 5}  # tid -> output dim

class MultiHeadResNet(nn.Module):
    def __init__(self, in_ch=7):
        super().__init__()
        self.backbone = models.resnet18(weights=None)

        # patch first conv
        old = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_ch, old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=old.bias is not None
        )

        # replace fc with identity, use our own heads
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.head1 = nn.Linear(feat_dim, 7)
        self.head2 = nn.Linear(feat_dim, 5)
        self.head3 = nn.Linear(feat_dim, 5)

    def forward(self, x, tid):
        feat = self.backbone(x)  # (B,feat)
        # allocate padded output (B,7)
        out = x.new_zeros((x.size(0), D_MAX))

        # select by tid (batch mixed)
        for t in (0, 1, 2):
            mask = (tid == t)
            if mask.any():
                f = feat[mask]
                if t == 0:
                    y = self.head1(f)               # (n,7)
                    out[mask, :7] = y
                elif t == 1:
                    y = self.head2(f)               # (n,5)
                    out[mask, :5] = y
                else:
                    y = self.head3(f)               # (n,5)
                    out[mask, :5] = y
        return out

def masked_mse(pred, y, mask):
    # pred,y,mask: (B,7)
    diff2 = (pred - y) ** 2
    diff2 = diff2 * mask
    denom = mask.sum().clamp_min(1.0)
    return diff2.sum() / denom

@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    total_count = 0

    # per type metrics (MAE on valid dims)
    mae_sum = {0: 0.0, 1: 0.0, 2: 0.0}
    mae_cnt = {0: 0.0, 1: 0.0, 2: 0.0}

    for x, y, m, tid, idx, path in loader:
        x, y, m, tid = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE), tid.to(DEVICE)
        pred = model(x, tid)
        loss = masked_mse(pred, y, m)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_count += bs

        abs_err = (pred - y).abs() * m
        for t in (0, 1, 2):
            mt = (tid == t)
            if mt.any():
                mae_sum[t] += abs_err[mt].sum().item()
                mae_cnt[t] += m[mt].sum().item()

    avg_loss = total_loss / max(1, total_count)
    mae = {t: (mae_sum[t] / max(1.0, mae_cnt[t])) for t in (0, 1, 2)}
    return avg_loss, mae

def main():
    torch.manual_seed(42)

    ds_all = ConcatDataset([CapRasterDataset("type1"), CapRasterDataset("type2"), CapRasterDataset("type3")])
    n = len(ds_all)
    n_train = int(n * 0.8)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds_all, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = MultiHeadResNet(in_ch=7).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val = float("inf")
    os.makedirs("ckpt", exist_ok=True)

    for epoch in range(1, 51):
        model.train()
        running = 0.0
        seen = 0

        for x, y, m, tid, idx, path in train_loader:
            x, y, m, tid = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE), tid.to(DEVICE)

            pred = model(x, tid)
            loss = masked_mse(pred, y, m)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / max(1, seen)
        val_loss, val_mae = eval_epoch(model, val_loader)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} "
              f"| val_mae t1={val_mae[0]:.6f} t2={val_mae[1]:.6f} t3={val_mae[2]:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, "ckpt/best.pt")
            print("  saved: ckpt/best.pt")

if __name__ == "__main__":
    main()
