import torch
from torch.utils.data import Subset, DataLoader, ConcatDataset
from step10_dataset_raster import CapRasterDataset

def get_type_id(ds):
    # ds[i] returns (x, y_pad, mask, tid, idx, path)
    return int(ds[i][3].item())

def main():
    torch.manual_seed(42)

    ds1 = CapRasterDataset("type1")
    ds2 = CapRasterDataset("type2")
    ds3 = CapRasterDataset("type3")

    # separate by type, stratified split inside each
    def split_one(ds, frac=0.8):
        n = len(ds)
        n_train = int(n * frac)
        perm = torch.randperm(n).tolist()
        tr = perm[:n_train]
        va = perm[n_train:]
        return tr, va

    tr1, va1 = split_one(ds1)
    tr2, va2 = split_one(ds2)
    tr3, va3 = split_one(ds3)

    # build concat, need offset indices
    off1 = 0
    off2 = len(ds1)
    off3 = len(ds1) + len(ds2)

    train_indices = [off1+i for i in tr1] + [off2+i for i in tr2] + [off3+i for i in tr3]
    val_indices   = [off1+i for i in va1] + [off2+i for i in va2] + [off3+i for i in va3]

    ds_all = ConcatDataset([ds1, ds2, ds3])
    train_ds = Subset(ds_all, train_indices)
    val_ds   = Subset(ds_all, val_indices)

    def count_types(subset):
        c = {0:0,1:0,2:0}
        for j in range(len(subset)):
            tid = int(subset[j][3].item())
            c[tid]+=1
        return c

    print("Total:", len(ds_all), "Train:", len(train_ds), "Val:", len(val_ds))
    print("Train type counts:", count_types(train_ds))
    print("Val   type counts:", count_types(val_ds))

if __name__ == "__main__":
    main()
