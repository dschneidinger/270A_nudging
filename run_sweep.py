"""Run all model types with multiple seeds and compare results.

Usage:
    python run_sweep.py

Each entry in EXPERIMENTS defines a model + config override. Conv1D2D is
tested at multiple timestep window sizes to give it a fair shot against
the flat models.
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

N_RUNS = 3  # seeds per experiment
NOISE_STDS = [0.1]

# Each experiment: a label (used for directory name + comparison chart)
# plus any config overrides.  Everything else comes from BASE_CONFIG.
EXPERIMENTS = [
    {"label": "mlp",              "model_type": "mlp"},
    {"label": "mlp_residual",     "model_type": "mlp_residual"},
    {"label": "fc_cnn",           "model_type": "fc_cnn"},
    {"label": "conv1d2d_nt5",     "model_type": "conv1d2d", "n_timesteps": 5},
    {"label": "conv1d2d_nt10",    "model_type": "conv1d2d", "n_timesteps": 10},
    {"label": "conv1d2d_nt20",    "model_type": "conv1d2d", "n_timesteps": 20},
    {"label": "attn_mlp_nt10",    "model_type": "attn_mlp", "n_timesteps": 10},
    {"label": "attn_mlp_nt20",    "model_type": "attn_mlp", "n_timesteps": 20},
]

BASE_CONFIG = {
    "data_path": "/Users/ARand/Desktop/270A_nudging/multiscale-nudging-main 2/case3_Vlasov_poisson_instability/simulation/data/mv_sim_seed0.npz",
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

    # Cache datasets by (input_mode, noise_std, n_timesteps) so we only
    # reload when the combination changes.
    dataset_cache = {}

    sweep_results = {}

    for exp in EXPERIMENTS:
        label = exp["label"]
        model_type = exp["model_type"]

        print(f"\n{'#' * 60}")
        print(f"# Experiment: {label}")
        print(f"{'#' * 60}")

        input_mode = get_input_mode(model_type)
        sweep_results[label] = {}

        for noise_std in NOISE_STDS:
            # Build config: base + experiment overrides
            config = {
                **BASE_CONFIG,
                "model_type": model_type,
                "n_runs": N_RUNS,
                "noise_std": noise_std,
                "device": str(device),
            }
            # Apply per-experiment overrides (e.g. n_timesteps)
            for k, v in exp.items():
                if k != "label":
                    config[k] = v

            n_timesteps = config["n_timesteps"]
            noise_tag = f"noise_{noise_std:.4f}"
            if sweeping_noise:
                print(f"\n  >> noise_std={noise_std}")

            # ── Data (cached) ──
            cache_key = (input_mode, noise_std, n_timesteps)
            if cache_key not in dataset_cache:
                print(f"Preparing data: input_mode={input_mode}, "
                      f"n_timesteps={n_timesteps}, noise_std={noise_std}")
                dataset_cache[cache_key] = prepare_training_data(
                    config["data_path"],
                    downsample_factor=config["downsample_factor"],
                    n_timesteps=n_timesteps,
                    target_offset=config["target_offset"],
                    input_mode=input_mode,
                    noise_std=noise_std,
                    train_frac=config["train_frac"],
                    val_frac=config["val_frac"],
                    test_frac=config["test_frac"],
                )
            else:
                print(f"Reusing cached data for {cache_key}")

            (train_dataset, val_dataset, test_dataset,
             n_sparse, n_x, n_v, norm_stats) = dataset_cache[cache_key]

            # ── Output dir ──
            if sweeping_noise:
                group_dir = os.path.join(sweep_dir, label, noise_tag)
            else:
                group_dir = os.path.join(sweep_dir, label)
            os.makedirs(group_dir, exist_ok=True)

            # Count params once
            tmp_model = build_model(model_type, n_sparse, n_x, n_v, config)
            n_params = sum(p.numel() for p in tmp_model.parameters())
            config["n_parameters"] = n_params
            print(f"Architecture: {label}  |  Parameters: {n_params:,}")
            del tmp_model

            with open(os.path.join(group_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

            # ── Run seeds ──
            seeds = list(range(N_RUNS))
            all_results = []

            for run_idx, seed in enumerate(seeds):
                run_dir = os.path.join(group_dir, f"run_{run_idx}_seed{seed}")
                os.makedirs(run_dir, exist_ok=True)

                tag = f"{label} noise={noise_std}" if sweeping_noise else label
                print(f"\n--- {tag} run {run_idx + 1}/{N_RUNS} (seed={seed}) ---")

                result = run_single_experiment(
                    model_type, config,
                    train_dataset, val_dataset, test_dataset,
                    n_sparse, n_x, n_v, norm_stats,
                    device, run_dir, seed,
                )
                all_results.append(result)

            # ── Per-experiment aggregated plot ──
            plot_aggregated_results(all_results, group_dir)

            # ── Collect summary stats ──
            test_mses = [r["test_mse"] for r in all_results]
            test_maes = [r["test_mae"] for r in all_results]
            test_ssims = [r["test_ssim"] for r in all_results]

            best_idx = int(np.argmin(test_mses))
            sweep_results[label][noise_std] = {
                "n_params": n_params,
                "noise_std": noise_std,
                "n_timesteps": n_timesteps,
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

    # ── Comparison plot ────────────────────────────────────────────────
    labels = [exp["label"] for exp in EXPERIMENTS]

    for noise_std in NOISE_STDS:
        flat = {lab: sweep_results[lab][noise_std] for lab in labels}
        plot_model_comparison(flat, sweep_dir)

        if sweeping_noise:
            src = os.path.join(sweep_dir, "model_comparison.png")
            dst = os.path.join(sweep_dir, f"model_comparison_noise{noise_std:.4f}.png")
            if os.path.exists(src):
                os.rename(src, dst)

    # ── Print summary table ────────────────────────────────────────────
    print(f"\n{'=' * 105}")
    print("SWEEP RESULTS")
    print(f"{'=' * 105}")

    for noise_std in NOISE_STDS:
        if sweeping_noise:
            print(f"\n  noise_std = {noise_std}")
            print(f"  {'-' * 100}")

        header = (f"  {'Experiment':<20} {'nt':>3} {'Params':>10} "
                  f"{'Test MSE':>22} {'Test MAE':>22} {'Test SSIM':>20}")
        print(header)
        print(f"  {'-' * 100}")
        for lab in labels:
            r = sweep_results[lab][noise_std]
            print(
                f"  {lab:<20} {r['n_timesteps']:>3} {r['n_params']:>10,} "
                f"{r['test_mse_mean']:>9.6f} +/- {r['test_mse_std']:<9.6f} "
                f"{r['test_mae_mean']:>9.6f} +/- {r['test_mae_std']:<9.6f} "
                f"{r['test_ssim_mean']:>7.4f} +/- {r['test_ssim_std']:<7.4f}"
            )

    print(f"\nAll artifacts in {sweep_dir}/")
