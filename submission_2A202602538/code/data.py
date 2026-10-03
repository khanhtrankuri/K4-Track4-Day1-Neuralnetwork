"""data.py — nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    train = np.load(f"{processed_dir}/train.npz")
    ev = np.load(f"{processed_dir}/eval.npz")

    X_train_full, y_train_full = train["X"], train["y"]
    X_eval, y_eval, eval_row_id = ev["X"], ev["y"], ev["row_id"]

    for X, y, tag in ((X_train_full, y_train_full, "train"), (X_eval, y_eval, "eval")):
        assert X.dtype == np.float32 and X.ndim == 2 and X.shape[1] == 54, \
            f"{tag}: X phải là float32 (N, 54), hiện {X.dtype} {X.shape}"
        assert y.dtype == np.int64 and y.ndim == 1 and y.shape[0] == X.shape[0], \
            f"{tag}: y phải là int64 (N,), hiện {y.dtype} {y.shape}"
        assert y.min() >= 0 and y.max() <= 6, f"{tag}: nhãn phải nằm trong 0..6"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed,
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Không được tính trên toàn bộ dữ liệu hay trên eval, vì khi đó thống kê sẽ "nhìn thấy" val/eval
    (rò rỉ thông tin), làm sai lệch đánh giá — mean/std phải chỉ phản ánh những gì model được huấn luyện trên.
    """
    numeric = X_tr[:, :N_NUMERIC].astype(np.float64)
    mean = numeric.mean(axis=0)
    std = numeric.std(axis=0)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X = X.copy()
    std_safe = np.where(std == 0, 1.0, std)  # cột hằng số (std=0): tránh chia 0, giữ nguyên giá trị đã trừ mean
    X[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std_safe
    return X


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction, seed)

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    device = torch.device(device)
    out = {
        "X_tr": torch.tensor(X_tr, dtype=torch.float32, device=device),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=device),
        "X_val": torch.tensor(X_val, dtype=torch.float32, device=device),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=device),
        "X_eval": torch.tensor(X_eval, dtype=torch.float32, device=device),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=device),
        "eval_row_id": eval_row_id,
    }

    majority_class = np.bincount(y_tr).argmax()
    majority_acc = float((y_val == majority_class).mean())
    print(f"X_tr   {tuple(out['X_tr'].shape)}   y_tr   {tuple(out['y_tr'].shape)}")
    print(f"X_val  {tuple(out['X_val'].shape)}   y_val  {tuple(out['y_val'].shape)}")
    print(f"X_eval {tuple(out['X_eval'].shape)}   y_eval {tuple(out['y_eval'].shape)}")
    print(f"luôn đoán lớp đa số (lớp {majority_class}) trên val -> accuracy = {majority_acc:.4f}")
    print(f"X_tr[:, :{N_NUMERIC}] mean = {X_tr[:, :N_NUMERIC].mean(0).round(3)}")
    print(f"X_tr[:, :{N_NUMERIC}] std  = {X_tr[:, :N_NUMERIC].std(0).round(3)}")
    return out


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Batch cuối có thể nhỏ hơn batch_size — ta GIỮ nó (không drop_last), vì bỏ 1 batch mỗi epoch
    thì mô hình không bao giờ thấy một số mẫu cố định và làm giảm nhẹ dữ liệu huấn luyện hiệu dụng.
    """
    n = X.shape[0]
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
