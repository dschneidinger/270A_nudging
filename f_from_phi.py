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
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

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
    """Dataset for training MLP to predict f from sparse phi measurements."""

    def __init__(self, phi_sparse, f_full, time_diffs):
        """
        Args:
            #TODO, I think you need to include the positions of the sparse measurements
            phi_sparse: (N_samples, 2, N_sparse) - sparse phi at t and t-1
            f_full: (N_samples, N_x, N_v) - full distribution function at time t
            time_diffs: (N_samples,) - time difference between measurements
        """
        self.phi_sparse = torch.FloatTensor(phi_sparse)
        self.f_full = torch.FloatTensor(f_full)
        self.time_diffs = torch.FloatTensor(time_diffs)

    def __len__(self):
        return len(self.phi_sparse)

    def __getitem__(self, idx):
        phi_t = self.phi_sparse[idx, 0, :]  # phi at time t
        dphi = self.phi_sparse[idx, 0, :] - self.phi_sparse[idx, 1, :]  # residual: phi(t) - phi(t-1)
        dt = self.time_diffs[idx : idx + 1]  # time difference

        # Concatenate: [phi_t, dphi, dt]
        input_vector = torch.cat([phi_t, dphi, dt])

        # Flatten the output f
        target = self.f_full[idx].flatten()

        return input_vector, target


class MLP(nn.Module):
    def __init__(
        self,
        n_sparse_measurements,
        n_x_grid,
        n_v_grid,
        hidden_layers=[256, 512, 1024, 512],
    ):
        """
        MLP to predict full distribution function from sparse potential measurements.

        Args:
            n_sparse_measurements: Number of sparse phi measurements per timestep
            n_x_grid: Number of spatial grid points (128)
            n_v_grid: Number of velocity grid points (64)
            hidden_layers: List of hidden layer sizes
        """
        super(MLP, self).__init__()

        # Input: sparse phi at t, sparse phi at t-1, and dt = 2*n_sparse + 1
        input_size = 2 * n_sparse_measurements + 1

        # Output: full f(x,v) distribution = n_x_grid * n_v_grid
        output_size = n_x_grid * n_v_grid

        self.n_x_grid = n_x_grid
        self.n_v_grid = n_v_grid

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (batch, 2*n_sparse + 1) - sparse phi measurements and time diff
        Returns:
            f: (batch, n_x_grid, n_v_grid) - predicted distribution function
        """
        out = self.network(x)
        # Reshape to (batch, n_x, n_v)
        return out.view(-1, self.n_x_grid, self.n_v_grid)


def prepare_training_data(
    data_path, downsample_factor=10, train_frac=0.80, val_frac=0.05, test_frac=0.15
):
    """
    Load and prepare training data from simulation output.

    Args:
        data_path: Path to .npz file with simulation data
        downsample_factor: Factor to downsample spatial measurements
        train_frac: Fraction of data for training (default 0.80)
        val_frac: Fraction of data for validation (default 0.05)
        test_frac: Fraction of data for testing (default 0.15)

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

    # Create training samples: use pairs of consecutive timesteps
    # Each sample: [phi(t), phi(t-1), dt] -> f(t)
    N_samples = len(times) - 1

    phi_input = np.zeros((N_samples, 2, n_sparse))  # (N_samples, 2, n_sparse)
    f_output = np.zeros((N_samples, n_x, n_v))  # (N_samples, n_x, n_v)
    time_diffs = np.zeros(N_samples)

    for i in range(N_samples):
        phi_input[i, 0, :] = phi_sparse[i + 1]  # phi at time t
        phi_input[i, 1, :] = phi_sparse[i]  # phi at time t-1
        f_output[i] = phase_density[i + 1]  # f at time t (ground truth)
        time_diffs[i] = times[i + 1] - times[i]  # dt

    # Random split into train / val / test
    rng = np.random.default_rng(42)
    indices = rng.permutation(N_samples)

    train_end = int(N_samples * train_frac)
    val_end = int(N_samples * (train_frac + val_frac))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    # Normalize inputs using training set statistics only
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
        phi_input[train_idx], f_output[train_idx], time_diffs[train_idx]
    )
    val_dataset = VlasovDataset(
        phi_input[val_idx], f_output[val_idx], time_diffs[val_idx]
    )
    test_dataset = VlasovDataset(
        phi_input[test_idx], f_output[test_idx], time_diffs[test_idx]
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
    Train the MLP model.

    Args:
        model: The MLP model
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


if __name__ == "__main__":
    # Configuration
    config = {
        "data_path": "/Users/ARand/Desktop/270A_nudging/multiscale-nudging-main 2/case3_Vlasov_poisson_instability/simulation/data/mv_sim_seed0.npz",
        "downsample_factor": 10,
        "batch_size": 32,
        "num_epochs": 100,
        "learning_rate": 0.001,
        "hidden_layers": [256, 512, 1024, 512, 256],
        "dropout": 0.1,
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

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join("experiments", timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = str(device)
    print(f"Using device: {device}")

    # Prepare data
    train_dataset, val_dataset, test_dataset, n_sparse, n_x, n_v, norm_stats = (
        prepare_training_data(
            config["data_path"],
            downsample_factor=config["downsample_factor"],
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
        "n_x": n_x,
        "n_v": n_v,
        "input_size": 2 * n_sparse + 1,
        "output_size": n_x * n_v,
    }
    with open(os.path.join(exp_dir, "split_info.json"), "w") as f:
        json.dump(split_info, f, indent=2)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=False
    )

    # Initialize model
    model = MLP(
        n_sparse_measurements=n_sparse,
        n_x_grid=n_x,
        n_v_grid=n_v,
        hidden_layers=config["hidden_layers"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    config["n_parameters"] = n_params
    print("Model architecture:")
    print(model)
    print(f"Total parameters: {n_params:,}")

    # Save model summary
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train the model
    print("Starting training...")
    train_losses, val_losses, val_mae_list, val_ssim_list = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=config["num_epochs"],
        lr=config["learning_rate"],
        device=device,
        exp_dir=exp_dir,
        norm_stats=norm_stats,
    )

    # Plot training history
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
        os.path.join(exp_dir, "training_history.png"), dpi=150, bbox_inches="tight"
    )
    print(f"Saved training history to {exp_dir}/training_history.png")

    # Evaluate on held-out test set using best model
    print("Test Set Evaluation")
    best_ckpt = torch.load(os.path.join(exp_dir, "best_model.pth"), weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    test_mse = 0.0
    test_mae = 0.0
    test_ssim = 0.0
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
    with open(os.path.join(exp_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)

    # Save final model
    torch.save(
        {
            "epoch": config["num_epochs"],
            "model_state_dict": model.state_dict(),
            "norm_stats": norm_stats,
        },
        os.path.join(exp_dir, "final_model.pth"),
    )
    print(f"Saved final model to {exp_dir}/final_model.pth")
    print(f"Experiment complete. All artifacts in {exp_dir}/")
    