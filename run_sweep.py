"""Run all model types with multiple seeds and compare results.

Usage:
    python run_sweep.py

Outputs go to experiments/<timestamp>_sweep/ with one subdirectory per
model type (and per noise level, if sweeping noise), each containing
per-seed run dirs plus an aggregated plot.
A final model_comparison.png and sweep_results.json summarize everything.
"""
import json
import os
from datetime import datetime

import torch
import numpy as np

from models import build_model, get_input_mode
from f_from_phi import prepare_training_data, run_single_experiment
from visualization import plot_aggregated_results, plot_model_comparison

# ── Configuration ────────────────────────────────────────────────────
# Edit these to control the sweep.

MODEL_TYPES = ["mlp", "mlp_residual", "fc_cnn", "conv1d2d"]
N_RUNS = 3  # seeds per model per noise level
NOISE_STDS = [0.1]  # noise stds to sweep

BASE_CONFIG = {
    "data_path": "multiscale-nudging-main/case3_Vlasov_poisson_instability/simulation/data/mv_sim_seed0.npz",
    "downsample_factor": 10,
    "n_timesteps": 2,
    "target_offset": 0,
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.001,
    # MLP
    "hidden_layers": [256, 512, 1024, 512, 256],
    "dropout": 0.1,
    # CNN
    "latent_dim": 256,
    # bookkeeping
    "activation": "SiLU",
    "loss": "MSE",
    "optimizer": "Adam",
    "scheduler": "ReduceLROnPlateau",
    "scheduler_factor": 0.5,
    "scheduler_patience": 5,
    "train_frac": 0.80,
    "val_frac": 0.05,
    "test_frac": 0.15,
}

# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    sweep_dir = os.path.join("experiments", f"{timestamp}_sweep")
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"Sweep directory: {sweep_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sweeping_noise = len(NOISE_STDS) > 1

    # Cache datasets by (input_mode, noise_std) so we only reload when
    # the combination changes.
    dataset_cache = {}

    # sweep_results is keyed by model_type.  Each entry holds a dict
    # keyed by noise_std (or a single entry when not sweeping noise).
    sweep_results = {}

    for model_type in MODEL_TYPES:
        print(f"\n{'#' * 60}")
        print(f"# Model: {model_type}")
        print(f"{'#' * 60}")

        input_mode = get_input_mode(model_type)
        sweep_results[model_type] = {}

        for noise_std in NOISE_STDS:
            config = {
                **BASE_CONFIG,
                "model_type": model_type,
                "n_runs": N_RUNS,
                "noise_std": noise_std,
                "device": str(device),
            }

            noise_tag = f"noise_{noise_std:.4f}"
            if sweeping_noise:
                print(f"\n  >> noise_std={noise_std}")

            # ── Data (cached by input_mode + noise_std) ──
            cache_key = (input_mode, noise_std)
            if cache_key not in dataset_cache:
                print(f"Preparing data for input_mode={input_mode}, noise_std={noise_std} ...")
                dataset_cache[cache_key] = prepare_training_data(
                    config["data_path"],
                    downsample_factor=config["downsample_factor"],
                    n_timesteps=config["n_timesteps"],
                    target_offset=config["target_offset"],
                    input_mode=input_mode,
                    noise_std=noise_std,
                    train_frac=config["train_frac"],
                    val_frac=config["val_frac"],
                    test_frac=config["test_frac"],
                )
            else:
                print(f"Reusing cached data for input_mode={input_mode}, noise_std={noise_std}")

            (train_dataset, val_dataset, test_dataset,
             n_sparse, n_x, n_v, norm_stats) = dataset_cache[cache_key]

            # ── Output dir ──
            if sweeping_noise:
                group_dir = os.path.join(sweep_dir, model_type, noise_tag)
            else:
                group_dir = os.path.join(sweep_dir, model_type)
            os.makedirs(group_dir, exist_ok=True)

            # Count params once
            tmp_model = build_model(model_type, n_sparse, n_x, n_v, config)
            n_params = sum(p.numel() for p in tmp_model.parameters())
            config["n_parameters"] = n_params
            print(f"Architecture: {model_type}  |  Parameters: {n_params:,}")
            del tmp_model

            with open(os.path.join(group_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            # ── Run seeds ──
            seeds = list(range(N_RUNS))
            all_results = []

            for run_idx, seed in enumerate(seeds):
                run_dir = os.path.join(group_dir, f"run_{run_idx}_seed{seed}")
                os.makedirs(run_dir, exist_ok=True)

                label = f"{model_type}"
                if sweeping_noise:
                    label += f" noise={noise_std}"
                print(f"\n--- {label} run {run_idx + 1}/{N_RUNS} (seed={seed}) ---")

                result = run_single_experiment(
                    model_type, config,
                    train_dataset, val_dataset, test_dataset,
                    n_sparse, n_x, n_v, norm_stats,
                    device, run_dir, seed,
                )
                all_results.append(result)

            # ── Per-group aggregated plot ──
            plot_aggregated_results(all_results, group_dir)

            # ── Collect summary stats ──
            test_mses = [r["test_mse"] for r in all_results]
            test_maes = [r["test_mae"] for r in all_results]
            test_ssims = [r["test_ssim"] for r in all_results]

            best_idx = int(np.argmin(test_mses))
            sweep_results[model_type][noise_std] = {
                "n_params": n_params,
                "noise_std": noise_std,
                "test_mse_mean": float(np.mean(test_mses)),
                "test_mse_std": float(np.std(test_mses)),
                "test_mae_mean": float(np.mean(test_maes)),
                "test_mae_std": float(np.std(test_maes)),
                "test_ssim_mean": float(np.mean(test_ssims)),
                "test_ssim_std": float(np.std(test_ssims)),
                "best_run": {
                    "index": best_idx,
                    "seed": all_results[best_idx]["seed"],
                    "test_mse": float(np.min(test_mses)),
                },
                "per_run": [
                    {
                        "seed": r["seed"],
                        "test_mse": r["test_mse"],
                        "test_mae": r["test_mae"],
                        "test_ssim": r["test_ssim"],
                        "best_epoch": r["best_epoch"],
                    }
                    for r in all_results
                ],
            }

    # ── Save full results ──────────────────────────────────────────────
    with open(os.path.join(sweep_dir, "sweep_results.json"), "w") as f:
        json.dump(sweep_results, f, indent=2)

    # ── Comparison plot (flatten for plot_model_comparison) ────────────
    # When sweeping noise, create one comparison chart per noise level.
    for noise_std in NOISE_STDS:
        flat = {}
        for m in MODEL_TYPES:
            r = sweep_results[m][noise_std]
            label = f"{m}" if not sweeping_noise else f"{m}"
            flat[label] = r

        suffix = "" if not sweeping_noise else f"_noise{noise_std:.4f}"
        sub_dir = sweep_dir  # save comparison charts in sweep root
        plot_model_comparison(flat, sub_dir)
        # rename if sweeping noise to avoid overwriting
        if sweeping_noise:
            src = os.path.join(sub_dir, "model_comparison.png")
            dst = os.path.join(sub_dir, f"model_comparison{suffix}.png")
            if os.path.exists(src):
                os.rename(src, dst)

    # ── Print summary table ────────────────────────────────────────────
    print(f"\n{'=' * 95}")
    print("SWEEP RESULTS")
    print(f"{'=' * 95}")

    for noise_std in NOISE_STDS:
        if sweeping_noise:
            print(f"\n  noise_std = {noise_std}")
            print(f"  {'-' * 90}")

        header = f"  {'Model':<15} {'Params':>10} {'Test MSE':>22} {'Test MAE':>22} {'Test SSIM':>20}"
        print(header)
        print(f"  {'-' * 90}")
        for m in MODEL_TYPES:
            r = sweep_results[m][noise_std]
            print(
                f"  {m:<15} {r['n_params']:>10,} "
                f"{r['test_mse_mean']:>9.6f} +/- {r['test_mse_std']:<9.6f} "
                f"{r['test_mae_mean']:>9.6f} +/- {r['test_mae_std']:<9.6f} "
                f"{r['test_ssim_mean']:>7.4f} +/- {r['test_ssim_std']:<7.4f}"
            )

    print(f"\nAll artifacts in {sweep_dir}/")
