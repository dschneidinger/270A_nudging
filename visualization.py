import os

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_training_history(train_losses, val_losses, val_mae_list, val_ssim_list, save_dir):
    """Plot per-run training curves (MSE, MAE, SSIM) and save to save_dir."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(train_losses, label="Train MSE")
    axes[0].plot(val_losses, label="Val MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title("MSE Loss")
    axes[0].grid(True)

    axes[1].plot(val_mae_list, label="Val MAE", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title("Mean Absolute Error")
    axes[1].grid(True)

    axes[2].plot(val_ssim_list, label="Val SSIM", color="green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].set_ylim(-0.1, 1.05)
    axes[2].legend()
    axes[2].set_title("Structural Similarity")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "training_history.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()


def plot_aggregated_results(all_results, exp_dir):
    """Plot mean +/- std training curves across multiple runs.

    Individual run traces are drawn with low opacity, the mean as a bold
    line, and the std as a shaded band.
    """
    n_runs = len(all_results)
    n_epochs = len(all_results[0]["train_losses"])
    epochs = np.arange(1, n_epochs + 1)

    train_mse = np.array([r["train_losses"] for r in all_results])
    val_mse = np.array([r["val_losses"] for r in all_results])
    val_mae = np.array([r["val_mae"] for r in all_results])
    val_ssim = np.array([r["val_ssim"] for r in all_results])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- MSE (log scale — clamp lower bound to stay positive) ---
    for i in range(n_runs):
        axes[0].plot(epochs, train_mse[i], color="tab:blue", alpha=0.15, lw=0.8)
        axes[0].plot(epochs, val_mse[i], color="tab:orange", alpha=0.15, lw=0.8)

    axes[0].plot(
        epochs, train_mse.mean(0), label="Train MSE (mean)",
        color="tab:blue", linewidth=2,
    )
    axes[0].fill_between(
        epochs,
        np.maximum(train_mse.mean(0) - train_mse.std(0), 1e-10),
        train_mse.mean(0) + train_mse.std(0),
        color="tab:blue", alpha=0.2,
    )
    axes[0].plot(
        epochs, val_mse.mean(0), label="Val MSE (mean)",
        color="tab:orange", linewidth=2,
    )
    axes[0].fill_between(
        epochs,
        np.maximum(val_mse.mean(0) - val_mse.std(0), 1e-10),
        val_mse.mean(0) + val_mse.std(0),
        color="tab:orange", alpha=0.2,
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title(f"MSE Loss ({n_runs} runs)")
    axes[0].grid(True)

    # --- MAE (log scale) ---
    for i in range(n_runs):
        axes[1].plot(epochs, val_mae[i], color="orange", alpha=0.15, lw=0.8)

    axes[1].plot(
        epochs, val_mae.mean(0), label="Val MAE (mean)",
        color="orange", linewidth=2,
    )
    axes[1].fill_between(
        epochs,
        np.maximum(val_mae.mean(0) - val_mae.std(0), 1e-10),
        val_mae.mean(0) + val_mae.std(0),
        color="orange", alpha=0.2,
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].set_title(f"Mean Absolute Error ({n_runs} runs)")
    axes[1].grid(True)

    # --- SSIM ---
    for i in range(n_runs):
        axes[2].plot(epochs, val_ssim[i], color="green", alpha=0.15, lw=0.8)

    axes[2].plot(
        epochs, val_ssim.mean(0), label="Val SSIM (mean)",
        color="green", linewidth=2,
    )
    axes[2].fill_between(
        epochs,
        val_ssim.mean(0) - val_ssim.std(0),
        val_ssim.mean(0) + val_ssim.std(0),
        color="green", alpha=0.2,
    )
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].set_ylim(-0.1, 1.05)
    axes[2].legend()
    axes[2].set_title(f"Structural Similarity ({n_runs} runs)")
    axes[2].grid(True)

    # Aggregate test metrics in suptitle
    test_mses = [r["test_mse"] for r in all_results]
    test_maes = [r["test_mae"] for r in all_results]
    test_ssims = [r["test_ssim"] for r in all_results]
    fig.suptitle(
        f"Test: MSE={np.mean(test_mses):.6f}\u00b1{np.std(test_mses):.6f}  "
        f"MAE={np.mean(test_maes):.6f}\u00b1{np.std(test_maes):.6f}  "
        f"SSIM={np.mean(test_ssims):.4f}\u00b1{np.std(test_ssims):.4f}",
        fontsize=11,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(exp_dir, "aggregated_training_history.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved aggregated plot to {exp_dir}/aggregated_training_history.png")


def plot_predictions(model, test_loader, n_x, n_v, device, save_dir, n_samples=6):
    """Plot predicted vs ground-truth phase-space f for a few test samples.

    For each sample draws three panels:
      Ground Truth | Prediction | Absolute Error

    Samples are spaced evenly across the test set so you see different
    regimes rather than just the first few.
    """
    model.eval()

    # Collect all test predictions
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # reshape to (B, n_x, n_v)
            all_preds.append(outputs.cpu().view(-1, n_x, n_v))
            all_targets.append(targets.cpu().view(-1, n_x, n_v))

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Pick evenly-spaced indices
    n_total = len(all_preds)
    n_samples = min(n_samples, n_total)
    indices = np.linspace(0, n_total - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        truth = all_targets[idx]
        pred = all_preds[idx]
        err = np.abs(pred - truth)

        # Shared color range for truth & prediction
        vmin = min(truth.min(), pred.min())
        vmax = max(truth.max(), pred.max())

        im0 = axes[row, 0].imshow(
            truth.T, aspect="auto", origin="lower", cmap="RdBu_r",
            vmin=vmin, vmax=vmax,
        )
        axes[row, 0].set_title(f"Ground Truth (sample {idx})")
        axes[row, 0].set_ylabel("v")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046, pad=0.04)

        im1 = axes[row, 1].imshow(
            pred.T, aspect="auto", origin="lower", cmap="RdBu_r",
            vmin=vmin, vmax=vmax,
        )
        axes[row, 1].set_title("Prediction")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046, pad=0.04)

        im2 = axes[row, 2].imshow(
            err.T, aspect="auto", origin="lower", cmap="hot",
        )
        axes[row, 2].set_title(f"|Error|  (max={err.max():.4f})")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046, pad=0.04)

    # Label bottom row x-axes
    for col in range(3):
        axes[-1, col].set_xlabel("x")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "predictions_vs_truth.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"Saved prediction comparison to {save_dir}/predictions_vs_truth.png")


def plot_model_comparison(sweep_results, save_dir):
    """Bar chart comparing test metrics (mean +/- std) across model types."""
    models = list(sweep_results.keys())
    n = len(models)
    x = np.arange(n)

    mse_mean = [sweep_results[m]["test_mse_mean"] for m in models]
    mse_std = [sweep_results[m]["test_mse_std"] for m in models]
    mae_mean = [sweep_results[m]["test_mae_mean"] for m in models]
    mae_std = [sweep_results[m]["test_mae_std"] for m in models]
    ssim_mean = [sweep_results[m]["test_ssim_mean"] for m in models]
    ssim_std = [sweep_results[m]["test_ssim_std"] for m in models]
    n_params = [sweep_results[m]["n_params"] for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"][:n]

    # MSE
    axes[0].bar(x, mse_mean, yerr=mse_std, color=colors, capsize=5, edgecolor="black")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=20, ha="right")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("MSE (lower is better)")
    axes[0].grid(axis="y", alpha=0.3)

    # MAE
    axes[1].bar(x, mae_mean, yerr=mae_std, color=colors, capsize=5, edgecolor="black")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=20, ha="right")
    axes[1].set_ylabel("Test MAE")
    axes[1].set_title("MAE (lower is better)")
    axes[1].grid(axis="y", alpha=0.3)

    # SSIM
    axes[2].bar(x, ssim_mean, yerr=ssim_std, color=colors, capsize=5, edgecolor="black")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=20, ha="right")
    axes[2].set_ylabel("Test SSIM")
    axes[2].set_title("SSIM (higher is better)")
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(axis="y", alpha=0.3)

    # Add param counts as text under each bar
    for ax in axes:
        for i, p in enumerate(n_params):
            ax.text(i, ax.get_ylim()[0], f"{p / 1e6:.1f}M", ha="center",
                    va="top", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, "model_comparison.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()
    print(f"Saved model comparison to {save_dir}/model_comparison.png")
