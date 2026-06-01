# Train a Network to find a distribution function f, from a given potential phi, measured sparsely in space.
import json
import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from models import build_model, get_input_mode
from visualization import plot_training_history, plot_aggregated_results, plot_predictions

r"""
Vlasov equation : \partial_t f + v \cdot \nabla_x f + F \cdot \nabla_v f = 0
For now we will just consider the electrostatic case, where F = q E = -\nabla_x * \phi * q #TODO check

Full distribution function f(x1,v1,t) is output from the numerical solver.
In the future this could be from osiris, but for now it will be from the numerical solver Hayden's postdoc wrote
"""


def phi_from_E(E, x_grid) -> np.ndarray:
    """Compute electric potential from electric field using integration."""
    phi = np.cumsum(-E * np.gradient(x_grid), axis=1)
    return phi


class VlasovDataset(Dataset):
    """Dataset for training a model to predict f from sparse phi measurements.

    Stores a sliding window of `n_timesteps` sparse phi measurements per sample.
    `input_mode` controls how that window is turned into the model input:

      - "flat_pair":     [phi_t, phi_{t-1}, dt]  (flat vector, size 2*N_sparse+1)
      - "flat_residual": [phi_t, phi_t-phi_{t-1}, dt]  (flat vector, size 2*N_sparse+1)
      - "sequence":      (n_timesteps+1, N_sparse) with a dt row appended

    For the flat modes only the last two timesteps of the window are used, so
    n_timesteps=2 exactly reproduces the original two-frame inputs.
    """

    def __init__(self, phi_sparse, f_full, time_diffs, input_mode="flat_pair"):
        """
        Args:
            phi_sparse: (N_samples, T, N_sparse) - T timesteps of sparse phi
            f_full: (N_samples, N_x, N_v) - full distribution function (target)
            time_diffs: (N_samples,) - time difference between measurements
            input_mode: one of "flat_pair", "flat_residual", "sequence"
        """
        self.phi_sparse = torch.FloatTensor(phi_sparse)
        self.f_full = torch.FloatTensor(f_full)
        self.time_diffs = torch.FloatTensor(time_diffs)
        self.input_mode = input_mode

    def __len__(self):
        return len(self.phi_sparse)

    def __getitem__(self, idx):
        phi_window = self.phi_sparse[idx]  # (T, N_sparse)
        dt = self.time_diffs[idx]

        if self.input_mode == "sequence":
            # Append dt as a constant row so the model sees the time spacing.
            dt_row = dt.expand(1, phi_window.shape[1])  # (1, N_sparse)
            inputs = torch.cat([phi_window, dt_row], dim=0)  # (T+1, N_sparse)
        else:
            # Flat modes use the last two timesteps of the window.
            # For flat_pair: channels are [phi_t, phi_{t-1}]
            # For flat_residual: channels are [phi_t, dphi] (dphi computed
            # on raw phi before normalization in prepare_training_data)
            phi_t = phi_window[-1]
            second_channel = phi_window[-2]
            dt_scalar = dt.reshape(1)
            inputs = torch.cat([phi_t, second_channel, dt_scalar])

        target = self.f_full[idx].flatten()
        return inputs, target


def prepare_training_data(
    data_path,
    downsample_factor=10,
    n_timesteps=2,
    target_offset=0,
    input_mode="flat_pair",
    noise_std=0.0,
    train_frac=0.80,
    val_frac=0.05,
    test_frac=0.15,
):
    """
    Load and prepare training data from simulation output.

    Builds sliding windows of `n_timesteps` sparse phi measurements. The target
    is f at index (window_end + target_offset):
      - target_offset=0 -> reconstruct f at the last timestep of the window
      - target_offset=1 -> forecast f one step past the window

    Args:
        data_path: Path to .npz file with simulation data
        downsample_factor: Factor to downsample spatial measurements
        n_timesteps: Number of timesteps per input window
        target_offset: Offset of target f relative to the window end (0 or 1)
        input_mode: How VlasovDataset builds inputs (set from model_type)
        noise_std: Std of Gaussian noise added to phi inputs (0 = clean)
        train_frac/val_frac/test_frac: Split fractions

    Returns:
        train_dataset, val_dataset, test_dataset, n_sparse, n_x, n_v, norm_stats
    """
    # Load the data
    data = np.load(data_path, allow_pickle=True)

    times = data["times"]  # (N_t,)
    electric = data["electric"]  # (N_t, N_x)
    phase_density = data["phase_density"]  # (N_t, N_x, N_v) - This is f!
    x_grid = data["x_grid"]  # (N_x,)

    print("Loaded data:")
    print(f"Times: {times.shape}")
    print(f"Electric field: {electric.shape}")
    print(f"Phase density (f): {phase_density.shape}")
    print(f"X grid: {x_grid.shape}")

    # Compute electric potential from electric field
    phi = phi_from_E(electric, x_grid)  # (N_t, N_x)

    # Downsample phi spatially
    phi_sparse = phi[:, ::downsample_factor]  # (N_t, N_x/downsample_factor)
    n_sparse = phi_sparse.shape[1]
    n_x, n_v = phase_density.shape[1], phase_density.shape[2]

    print(f"Sparse phi measurements: {n_sparse} per timestep")
    print(f"Output grid: {n_x} x {n_v} = {n_x * n_v} values")

    # Create training samples: sliding window of n_timesteps.
    # Window i covers phi_sparse[i : i+n_timesteps]; window end index is
    # i + n_timesteps - 1. Target f is at (window_end + target_offset).
    N_samples = len(times) - n_timesteps

    phi_input = np.zeros((N_samples, n_timesteps, n_sparse))
    f_output = np.zeros((N_samples, n_x, n_v))
    time_diffs = np.zeros(N_samples)

    for i in range(N_samples):
        phi_input[i] = phi_sparse[i : i + n_timesteps]  # window of T timesteps
        target_idx = i + n_timesteps - 1 + target_offset
        f_output[i] = phase_density[target_idx]
        time_diffs[i] = times[i + 1] - times[i]  # dt (constant spacing)

    # Add Gaussian noise to phi inputs (simulates noisy observations)
    if noise_std > 0:
        noise_rng = np.random.default_rng(123)
        phi_input += noise_std * noise_rng.standard_normal(phi_input.shape)
        print(f"Added Gaussian noise to phi inputs (std={noise_std})")

    # Random split into train / val / test
    rng = np.random.default_rng(42)
    indices = rng.permutation(N_samples)

    train_end = int(N_samples * train_frac)
    val_end = int(N_samples * (train_frac + val_frac))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # For residual mode, replace the second channel with dphi = phi_t - phi_{t-1}
    # BEFORE normalization so that dphi gets its own scale instead of being a
    # near-zero difference of two independently standardized signals.
    if input_mode == "flat_residual":
        phi_t = phi_input[:, -1, :]       # (N, n_sparse)
        phi_t_minus_1 = phi_input[:, -2, :]
        dphi = phi_t - phi_t_minus_1      # raw residual
        phi_input[:, -2, :] = dphi        # overwrite second channel with dphi

    # Normalize inputs using training set statistics only
    # (each channel now has its own meaningful scale — phi_t vs dphi for residual)
    phi_mean = phi_input[train_idx].mean(axis=0)
    phi_std = phi_input[train_idx].std(axis=0) + 1e-8
    phi_input = (phi_input - phi_mean) / phi_std

    dt_mean = time_diffs[train_idx].mean()
    dt_std = time_diffs[train_idx].std() + 1e-8
    time_diffs = (time_diffs - dt_mean) / dt_std

    # Normalize outputs using training set statistics only
    f_mean = f_output[train_idx].mean()
    f_std = f_output[train_idx].std() + 1e-8
    f_output = (f_output - f_mean) / f_std

    train_dataset = VlasovDataset(
        phi_input[train_idx], f_output[train_idx], time_diffs[train_idx], input_mode
    )
    val_dataset = VlasovDataset(
        phi_input[val_idx], f_output[val_idx], time_diffs[val_idx], input_mode
    )
    test_dataset = VlasovDataset(
        phi_input[test_idx], f_output[test_idx], time_diffs[test_idx], input_mode
    )

    print("Dataset split:")
    print(f"Train: {len(train_dataset)}")
    print(f"Val: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")

    # Return normalization stats so predictions can be un-normalized at inference
    norm_stats = {
        "phi_mean": phi_mean,
        "phi_std": phi_std,
        "dt_mean": dt_mean,
        "dt_std": dt_std,
        "f_mean": f_mean,
        "f_std": f_std,
    }

    return train_dataset, val_dataset, test_dataset, n_sparse, n_x, n_v, norm_stats


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    lr=0.001,
    device="cpu",
    exp_dir=None,
    norm_stats=None,
):
    """
    Train the model.

    Args:
        model: The model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
        exp_dir: Experiment directory for saving logs and checkpoints
        norm_stats: Normalization statistics to save with checkpoints
    """
    model = model.to(device)

    # Use MSE loss for regression (not CrossEntropyLoss which is for classification!)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    train_losses = []
    val_losses = []
    val_mae_list = []
    val_ssim_list = []

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    # Set up CSV log
    csv_path = os.path.join(exp_dir, "training_log.csv") if exp_dir else None
    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_mse", "val_mse", "val_mae", "val_ssim", "lr"]
            )

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_flat = outputs.view(outputs.size(0), -1)
            loss = criterion(outputs_flat, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_ssim = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs_flat = outputs.view(outputs.size(0), -1)
                targets_2d = targets.view(-1, model.n_x_grid, model.n_v_grid)
                loss = criterion(outputs_flat, targets)
                val_loss += loss.item()
                val_mae += F.l1_loss(outputs_flat, targets).item()
                # SSIM expects (B, C, H, W) — add channel dim
                val_ssim += ssim_metric(
                    outputs.unsqueeze(1), targets_2d.unsqueeze(1)
                ).item()
                n_val_batches += 1

        val_loss /= n_val_batches
        val_mae /= n_val_batches
        val_ssim /= n_val_batches
        val_losses.append(val_loss)
        val_mae_list.append(val_mae)
        val_ssim_list.append(val_ssim)

        current_lr = optimizer.param_groups[0]["lr"]

        # Log to CSV
        if csv_path:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [epoch + 1, train_loss, val_loss, val_mae, val_ssim, current_lr]
                )

        # Save best model checkpoint
        if exp_dir and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "norm_stats": norm_stats,
                },
                os.path.join(exp_dir, "best_model.pth"),
            )

        # Update learning rate
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}]")
            print(f"Train MSE: {train_loss:.6f}")
            print(f"Val MSE: {val_loss:.6f}")
            print(f"Val MAE: {val_mae:.6f}")
            print(f"Val SSIM: {val_ssim:.4f}")
            print(f"LR: {current_lr:.2e}")

    return train_losses, val_losses, val_mae_list, val_ssim_list


def run_single_experiment(
    model_type,
    config,
    train_dataset,
    val_dataset,
    test_dataset,
    n_sparse,
    n_x,
    n_v,
    norm_stats,
    device,
    run_dir,
    seed,
):
    """Run one training experiment with a given random seed.

    Data split is fixed; the seed controls model initialization and
    training shuffle order.

    Returns a dict of per-epoch training curves and test-set metrics.
    """
    # Set seeds for reproducibility of this run
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Use a seeded generator so DataLoader shuffle is reproducible per run
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, generator=g
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Build a fresh model (random weight init depends on the seed)
    model = build_model(model_type, n_sparse, n_x, n_v, config)

    # Train
    train_losses, val_losses, val_mae_list, val_ssim_list = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config["num_epochs"],
        lr=config["learning_rate"],
        device=device,
        exp_dir=run_dir,
        norm_stats=norm_stats,
    )

    plot_training_history(train_losses, val_losses, val_mae_list, val_ssim_list, run_dir)

    # Evaluate on test set using best model
    print("Test Set Evaluation")
    best_ckpt = torch.load(
        os.path.join(run_dir, "best_model.pth"), weights_only=False
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    test_mse, test_mae, test_ssim = 0.0, 0.0, 0.0
    n_test_batches = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs_flat = outputs.view(outputs.size(0), -1)
            targets_2d = targets.view(-1, model.n_x_grid, model.n_v_grid)
            test_mse += F.mse_loss(outputs_flat, targets).item()
            test_mae += F.l1_loss(outputs_flat, targets).item()
            test_ssim += ssim_metric(
                outputs.unsqueeze(1), targets_2d.unsqueeze(1)
            ).item()
            n_test_batches += 1

    test_mse /= n_test_batches
    test_mae /= n_test_batches
    test_ssim /= n_test_batches

    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Test SSIM: {test_ssim:.4f}")

    test_results = {"test_mse": test_mse, "test_mae": test_mae, "test_ssim": test_ssim}
    with open(os.path.join(run_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    # Prediction vs ground-truth visualizations (model is already best-ckpt)
    plot_predictions(model, test_loader, n_x, n_v, device, run_dir)

    # Save final model
    torch.save(
        {
            "epoch": config["num_epochs"],
            "model_state_dict": model.state_dict(),
            "norm_stats": norm_stats,
        },
        os.path.join(run_dir, "final_model.pth"),
    )
    print(f"Saved models to {run_dir}/")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_mae": val_mae_list,
        "val_ssim": val_ssim_list,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_ssim": test_ssim,
        "best_val_loss": best_ckpt["val_loss"],
        "best_epoch": best_ckpt["epoch"],
        "seed": seed,
    }


if __name__ == "__main__":
    # Configuration
    #
    # Pick the experiment with `model_type`. Keys are grouped by which models
    # use them; unused keys are simply ignored by the other models.
    config = {
        "data_path": "multiscale-nudging-main/case3_Vlasov_poisson_instability/simulation/data/mv_sim_seed0.npz",
        # --- experiment selection ---
        "model_type": "mlp",  # one of: mlp, mlp_residual, fc_cnn, conv1d2d
        "downsample_factor": 10,
        "n_timesteps": 2,  # input window length (flat models use last 2)
        "target_offset": 0,  # 0 = reconstruct f at window end; 1 = forecast +1 step
        # --- training ---
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 0.001,
        # --- multi-run averaging ---
        "n_runs": 1,  # number of seeds to run; >1 averages results
        "noise_std": 0.0,  # Gaussian noise std on phi inputs (0 = clean)
        # --- MLP (mlp / mlp_residual) only ---
        "hidden_layers": [256, 512, 1024, 512, 256],
        "dropout": 0.1,
        # --- fc_cnn / conv1d2d only ---
        "latent_dim": 256,
        # --- bookkeeping ---
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

    model_type = config["model_type"]
    input_mode = get_input_mode(model_type)
    n_runs = config.get("n_runs", 1)
    print(f"Experiment: model_type={model_type} (input_mode={input_mode}), n_runs={n_runs}")

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("experiments", f"{timestamp}_{model_type}")
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)
    print(f"Using device: {device}")

    # Prepare data once — the split is deterministic (seed=42 inside
    # prepare_training_data) so every run sees the same train/val/test split.
    train_dataset, val_dataset, test_dataset, n_sparse, n_x, n_v, norm_stats = (
        prepare_training_data(
            config["data_path"],
            downsample_factor=config["downsample_factor"],
            n_timesteps=config["n_timesteps"],
            target_offset=config["target_offset"],
            input_mode=input_mode,
            noise_std=config.get("noise_std", 0.0),
            train_frac=config["train_frac"],
            val_frac=config["val_frac"],
            test_frac=config["test_frac"],
        )
    )

    # Save split info
    split_info = {
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "n_test": len(test_dataset),
        "n_sparse": n_sparse,
        "n_timesteps": config["n_timesteps"],
        "n_x": n_x,
        "n_v": n_v,
        "output_size": n_x * n_v,
    }
    with open(os.path.join(exp_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    # Log model architecture once (architecture is the same across seeds)
    tmp_model = build_model(model_type, n_sparse, n_x, n_v, config)
    n_params = sum(p.numel() for p in tmp_model.parameters())
    config["n_parameters"] = n_params
    print("Model architecture:")
    print(tmp_model)
    print(f"Total parameters: {n_params:,}")
    del tmp_model

    # Re-save config with n_parameters
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ---- Run experiment(s) ----
    seeds = list(range(n_runs))
    all_results = []

    for run_idx, seed in enumerate(seeds):
        if n_runs > 1:
            run_dir = os.path.join(exp_dir, f"run_{run_idx}_seed{seed}")
            os.makedirs(run_dir, exist_ok=True)
        else:
            run_dir = exp_dir

        print(f"\n{'=' * 60}")
        print(f"Run {run_idx + 1}/{n_runs} (seed={seed})")
        print(f"{'=' * 60}")

        result = run_single_experiment(
            model_type,
            config,
            train_dataset,
            val_dataset,
            test_dataset,
            n_sparse,
            n_x,
            n_v,
            norm_stats,
            device,
            run_dir,
            seed,
        )
        all_results.append(result)

    # ---- Aggregate results (multi-run only) ----
    if n_runs > 1:
        plot_aggregated_results(all_results, exp_dir)

        test_mses = [r["test_mse"] for r in all_results]
        test_maes = [r["test_mae"] for r in all_results]
        test_ssims = [r["test_ssim"] for r in all_results]

        best_idx = int(np.argmin(test_mses))
        summary = {
            "n_runs": n_runs,
            "seeds": seeds,
            "per_run": [
                {
                    "run": i,
                    "seed": r["seed"],
                    "test_mse": r["test_mse"],
                    "test_mae": r["test_mae"],
                    "test_ssim": r["test_ssim"],
                    "best_val_loss": r["best_val_loss"],
                    "best_epoch": r["best_epoch"],
                }
                for i, r in enumerate(all_results)
            ],
            "aggregate": {
                "test_mse_mean": float(np.mean(test_mses)),
                "test_mse_std": float(np.std(test_mses)),
                "test_mae_mean": float(np.mean(test_maes)),
                "test_mae_std": float(np.std(test_maes)),
                "test_ssim_mean": float(np.mean(test_ssims)),
                "test_ssim_std": float(np.std(test_ssims)),
            },
            "best_run": {
                "index": best_idx,
                "seed": all_results[best_idx]["seed"],
                "test_mse": float(np.min(test_mses)),
            },
        }
        with open(os.path.join(exp_dir, "aggregate_results.json"), "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"Aggregate Results ({n_runs} runs)")
        print(f"{'=' * 60}")
        print(f"Test MSE:  {np.mean(test_mses):.6f} +/- {np.std(test_mses):.6f}")
        print(f"Test MAE:  {np.mean(test_maes):.6f} +/- {np.std(test_maes):.6f}")
        print(f"Test SSIM: {np.mean(test_ssims):.4f} +/- {np.std(test_ssims):.4f}")
        print(
            f"Best run: #{best_idx} seed={summary['best_run']['seed']} "
            f"(MSE={summary['best_run']['test_mse']:.6f})"
        )

    print(f"\nExperiment complete. All artifacts in {exp_dir}/")
