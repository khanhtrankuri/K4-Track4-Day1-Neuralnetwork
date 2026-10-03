"""train.py — đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.

Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.

Quy ước đã chọn (ghi vào báo cáo):
  - MSE: tổng bình phương sai số trên 7 logit của MỘT mẫu, rồi lấy trung bình theo mẫu
    (tức = 7 * F.mse_loss mặc định). Như vậy loss "trên mỗi mẫu" có cùng ý nghĩa với CE.
  - train_loss cuối epoch tính trên một tập con CỐ ĐỊNH 50 000 mẫu của train (chọn 1 lần, seed 0).
  - epoch_time_s chỉ đo vòng cập nhật tham số (không tính thời gian evaluate cuối epoch).
  - grad_norm là chuẩn L2 toàn cục TRƯỚC khi cắt, lấy trung bình theo các bước trong epoch.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

N_CLASSES = 7
TRAIN_LOSS_SUBSET = 50_000


def _find_repo_root() -> Path:
    """Dò lên từ vị trí file này tới thư mục chứa `scripts/evaluate.py`.

    `train.py` có thể nằm ở `code/` ngay dưới repo root (lúc phát triển) hoặc ở
    `submission_<MSSV>/code/` (sau khi nộp bài, lệch thêm một cấp) — dò động để đúng cả hai trường hợp.
    """
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "scripts" / "evaluate.py").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Không tìm thấy scripts/evaluate.py ở các thư mục cha của train.py")


REPO_ROOT = _find_repo_root()


# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=0.001,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0.

    cm: ma trận nhầm lẫn (7, 7), hàng = nhãn thật, cột = dự đoán.
    Giống hệt scripts/evaluate.py (không dùng epsilon, chia an toàn bằng np.divide(where=...)).
    """
    cm = np.asarray(cm)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    model.eval()
    preds = []
    for i in range(0, X.size(0), batch_size):
        preds.append(model(X[i:i + batch_size]).float().argmax(dim=1))
    return torch.cat(preds)


def compute_loss(logits, y, loss_name: str, reduction: str = "mean"):
    """"ce"  : cross-entropy nhận logit thô và nhãn int64 (F.cross_entropy).
       "mse" : tổng bình phương (logit - one_hot(y)) trên 7 lớp của mỗi mẫu, rồi mean/sum theo mẫu.
    reduction: "mean" (dùng khi huấn luyện) | "sum" (dùng trong evaluate để cộng dồn theo lô).
    """
    if loss_name == "ce":
        return F.cross_entropy(logits, y, reduction=reduction)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=logits.size(1)).to(logits.dtype)
        per_sample = ((logits - target) ** 2).sum(dim=1)
        return per_sample.mean() if reduction == "mean" else per_sample.sum()
    raise ValueError(f"Unknown loss name: {loss_name}")


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad, tính bằng fp32."""
    model.eval()
    n = X.size(0)
    total_loss = 0.0
    preds = []
    for i in range(0, n, batch_size):
        logits = model(X[i:i + batch_size]).float()
        total_loss += compute_loss(logits, y[i:i + batch_size], loss_name, reduction="sum").item()
        preds.append(logits.argmax(dim=1))
    pred = torch.cat(preds)

    acc = (pred == y).float().mean().item()
    # cm[thật, dự đoán] — đếm bằng bincount trên chỉ số phẳng
    cm = torch.bincount(y * N_CLASSES + pred, minlength=N_CLASSES * N_CLASSES)
    cm = cm.reshape(N_CLASSES, N_CLASSES).cpu().numpy()
    return {"loss": total_loss / n, "acc": acc, "macro_f1": macro_f1_from_confusion(cm)}


def _build_model(cfg: dict, in_features: int, device) -> MLP:
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=cfg["dropout"], init=cfg["init"], in_features=in_features)
    n_params = count_params(model)
    assert n_params == EXPECTED_PARAMS[hidden], \
        f"Số tham số {n_params} != {EXPECTED_PARAMS[hidden]} cho hidden={hidden}"
    return model.to(device)


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt.

    Args:
        cfg : dict cấu hình (xem DEFAULT_CFG)
        data: kết quả của data.prepare_data (tensor X_tr, y_tr, X_val, y_val, X_eval, y_eval trên device)

    Trả về dict:
        {"cfg": cfg,
         "history": {"epoch": [...], "train_loss": [...], "val_loss": [...], "val_acc": [...],
                     "val_macro_f1": [...], "grad_norm": [...], "epoch_time_s": [...]},
         "summary": {"step0_loss", "best_val_loss", "best_epoch", "final_train_loss", "final_val_loss",
                     "val_acc", "val_macro_f1", "time_per_epoch_s", "peak_mem_MB", "diverged"},
         "best_state": state_dict của epoch có val_loss thấp nhất (giữ trong RAM để dự đoán eval)}

    TUYỆT ĐỐI không đưa X_eval vào hàm này để chọn epoch/cấu hình. Chỉ dùng val.
    """
    cfg = {**DEFAULT_CFG, **cfg}
    X_tr, y_tr = data["X_tr"], data["y_tr"]
    X_val, y_val = data["X_val"], data["y_val"]
    device = X_tr.device
    use_cuda = device.type == "cuda"
    loss_name = cfg["loss"]
    precision = cfg["precision"]
    if precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"Unknown precision: {precision}")

    # ---- 0. seed, model, optimizer, scaler
    set_seed(cfg["seed"])
    model = _build_model(cfg, X_tr.shape[1], device)
    opt = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"],
                          weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))
    # generator riêng cho việc xáo dữ liệu -> thứ tự batch chỉ phụ thuộc seed
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg["seed"])

    # tập con CỐ ĐỊNH của train để đo train_loss (giống nhau giữa mọi thí nghiệm)
    n_sub = min(TRAIN_LOSS_SUBSET, X_tr.size(0))
    sub_idx = torch.randperm(X_tr.size(0), generator=torch.Generator().manual_seed(0))[:n_sub].to(device)
    X_tr_sub, y_tr_sub = X_tr[sub_idx], y_tr[sub_idx]

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    # ---- 1. loss tại bước 0 (trước cập nhật đầu tiên); với CE kỳ vọng ≈ ln 7 ≈ 1.946
    step0_loss = evaluate(model, X_val, y_val, loss_name)["loss"]

    history = {k: [] for k in ("epoch", "train_loss", "val_loss", "val_acc",
                               "val_macro_f1", "grad_norm", "epoch_time_s")}
    best_val_loss, best_epoch, best_state = math.inf, None, None
    diverged = False

    # ---- 2. vòng huấn luyện
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        grad_norms = []
        if use_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        for xb, yb in iterate_batches(X_tr, y_tr, cfg["batch"], gen):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(xb)
                loss = compute_loss(logits.float(), yb, loss_name)

            if not torch.isfinite(loss).item():
                diverged = True
                break

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()          # scaler bị tắt (no-op) khi không phải fp16
            scaler.unscale_(opt)                   # đưa grad về thang thật TRƯỚC khi đo/cắt
            gn = clip_gradients(model.parameters(), cfg["clip_norm"])
            grad_norms.append(gn)
            scaler.step(opt)                       # = opt.step() khi scaler tắt
            scaler.update()

        if use_cuda:
            torch.cuda.synchronize(device)
        epoch_time = time.perf_counter() - t0

        if diverged:
            print(f"[{cfg['exp_id']}] loss NaN/inf ở epoch {epoch} -> dừng sớm")
            break

        tr = evaluate(model, X_tr_sub, y_tr_sub, loss_name)
        va = evaluate(model, X_val, y_val, loss_name)
        finite_gn = [g for g in grad_norms if math.isfinite(g)]
        history["epoch"].append(epoch)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_acc"].append(va["acc"])
        history["val_macro_f1"].append(va["macro_f1"])
        history["grad_norm"].append(float(np.mean(finite_gn)) if finite_gn else math.nan)
        history["epoch_time_s"].append(epoch_time)

        print(f"[{cfg['exp_id']}] ep {epoch:3d} | train {tr['loss']:.4f} | val {va['loss']:.4f} "
              f"acc {va['acc']:.4f} f1 {va['macro_f1']:.4f} | gn {history['grad_norm'][-1]:.3f} "
              f"| {epoch_time:.1f}s")

        if not math.isfinite(va["loss"]):
            diverged = True
            print(f"[{cfg['exp_id']}] val_loss NaN/inf ở epoch {epoch} -> dừng sớm")
            break

        if va["loss"] < best_val_loss:
            best_val_loss, best_epoch = va["loss"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # ---- 3. tóm tắt tại best_epoch
    if best_epoch is not None:
        bi = history["epoch"].index(best_epoch)
        val_acc, val_f1 = history["val_acc"][bi], history["val_macro_f1"][bi]
    else:
        val_acc = val_f1 = math.nan
    summary = {
        "step0_loss": step0_loss,
        "best_val_loss": best_val_loss if best_epoch is not None else math.nan,
        "best_epoch": best_epoch,
        "final_train_loss": history["train_loss"][-1] if history["epoch"] else math.nan,
        "final_val_loss": history["val_loss"][-1] if history["epoch"] else math.nan,
        "val_acc": val_acc,
        "val_macro_f1": val_f1,
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch"] else math.nan,
        "peak_mem_MB": torch.cuda.max_memory_allocated(device) / 2**20 if use_cuda else math.nan,
        "diverged": diverged,
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`.

    row_id : mảng row_id của tập eval (data["eval_row_id"])
    preds  : nhãn dự đoán int64 0..6 (cùng thứ tự với row_id)
    Phải đủ mọi dòng của tập eval, mỗi row_id đúng một lần.
    """
    row_id = np.asarray(row_id).astype(np.int64).ravel()
    preds = np.asarray(preds).astype(np.int64).ravel()
    assert len(row_id) == len(preds), f"len(row_id)={len(row_id)} != len(preds)={len(preds)}"
    assert len(np.unique(row_id)) == len(row_id), "row_id bị lặp"
    assert preds.min() >= 0 and preds.max() <= N_CLASSES - 1, "pred phải nằm trong 0..6"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack([row_id, preds]), fmt="%d", delimiter=",",
               header="row_id,pred", comments="")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str, out_json: str | None = None) -> dict:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions,
    rồi chạy scripts/evaluate.py và in kết quả (ghi kèm JSON cạnh file dự đoán).

    out_json: đường dẫn JSON kết quả; mặc định "eval_result.json" cùng thư mục với pred_path
             (đúng như lệnh gợi ý trong lab.ipynb).
    Trả về dict đọc từ out_json (accuracy, macro_f1, per_class, confusion_matrix) để ghi vào bảng/báo cáo.
    """
    if result.get("best_state") is None:
        raise ValueError("result không có best_state (thí nghiệm phân kỳ trước epoch đầu?)")
    cfg = {**DEFAULT_CFG, **cfg}
    device = data["X_eval"].device

    # 1. dựng lại model và nạp trọng số tốt nhất
    model = _build_model(cfg, data["X_eval"].shape[1], device)
    model.load_state_dict(result["best_state"])

    # 2. dự đoán eval ở fp32, eval mode
    preds = predict(model, data["X_eval"])

    # 3. ghi file nộp
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
    print(f"đã ghi {pred_path} ({len(preds)} dòng)")

    # 4. chấm bằng script chính thức (chạy từ thư mục gốc repo để đúng đường dẫn data/)
    pred_abs = Path(pred_path).resolve()
    out_json_abs = Path(out_json).resolve() if out_json is not None else pred_abs.with_name("eval_result.json")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "evaluate.py"),
         "--pred", str(pred_abs), "--out", str(out_json_abs)],
        cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8",
    )
    print(proc.stdout)
    if proc.returncode != 0:
        print(proc.stderr)
        raise RuntimeError("scripts/evaluate.py báo lỗi — xem output ở trên")

    with open(out_json_abs, encoding="utf-8") as f:
        return json.load(f)
