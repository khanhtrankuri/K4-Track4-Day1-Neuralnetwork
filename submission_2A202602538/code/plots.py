"""plots.py — vẽ biểu đồ cho từng thí nghiệm và biểu đồ so sánh nhiều thí nghiệm.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def _cfg_subtitle(cfg: dict) -> str:
    hidden = cfg.get("hidden")
    hidden_str = "-".join(str(h) for h in hidden) if hidden is not None else "?"
    return (f"{cfg.get('optimizer')} | lr={cfg.get('lr')} | batch={cfg.get('batch')} | "
            f"hidden={hidden_str} | loss={cfg.get('loss')} | seed={cfg.get('seed')}")


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Đánh dấu best_epoch bằng đường thẳng đứng trên cả 3 ô.
    """
    cfg = result["cfg"]
    history = result["history"]
    summary = result["summary"]
    epochs = history["epoch"]
    best_epoch = summary.get("best_epoch")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train_loss", marker=".")
    ax.plot(epochs, history["val_loss"], label="val_loss", marker=".")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["val_acc"], label="val_acc", marker=".")
    ax.plot(epochs, history["val_macro_f1"], label="val_macro_f1", marker=".")
    ax.set_xlabel("epoch")
    ax.set_ylabel("score")
    ax.set_title("Validation metrics")
    ax.legend()

    ax = axes[2]
    ax.plot(epochs, history["grad_norm"], label="grad_norm (trước clip)", marker=".", color="tab:red")
    ax.set_xlabel("epoch")
    ax.set_ylabel("grad norm (L2)")
    ax.set_title("Gradient norm")
    ax.legend()

    if best_epoch is not None:
        for ax in axes:
            ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                       label=f"best_epoch={best_epoch}")
            ax.legend()

    fig.suptitle(f"{cfg.get('exp_id')} — {cfg.get('description', '')}\n{_cfg_subtitle(cfg)}")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for result in results:
        history = result["history"]
        exp_id = result["cfg"].get("exp_id", "?")
        if metric not in history:
            raise KeyError(f"'{metric}' không có trong history của {exp_id}")
        ax.plot(history["epoch"], history[metric], marker=".", label=exp_id)

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.set_title(title or f"So sánh {metric}")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
