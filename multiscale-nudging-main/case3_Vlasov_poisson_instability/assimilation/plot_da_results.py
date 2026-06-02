"""Plot diagnostics for a phase-space KDE-nudging DA run.

Reads the .npz written by simulation.py and produces:
  - error_curves.png        phase-space RMSE to truth / obs vs time
  - phase_space_snapshots.png  true / estimate / (NN target) density at a few times
  - field_diagnostics.png   E-field mode-1 amplitude (truth vs estimate) over time

Usage:
    python plot_da_results.py <result.npz> [--outdir <dir>]
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_result(path):
    data = np.load(path, allow_pickle=True)
    return data


def plot_error_curves(data, outdir):
    times = data["times"]
    err_truth = data["phase_err_truth"]
    err_obs = data["phase_err_obs"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, err_truth, label="RMSE to truth", lw=1.5)
    # phase_err_obs can be NaN on fully-missing steps; mask those.
    finite = np.isfinite(err_obs)
    ax.plot(times[finite], err_obs[finite], label="RMSE to obs density", lw=1.0, alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("phase-space density RMSE")
    ax.set_yscale("log")
    ax.set_title("Nudging error vs time")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, "error_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _imshow_density(ax, density, x_mesh, v_mesh, title, vmin=None, vmax=None):
    extent = [v_mesh[0], v_mesh[-1], x_mesh[0], x_mesh[-1]]
    im = ax.imshow(
        density,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="viridis",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("v")
    ax.set_ylabel("x")
    return im


def plot_phase_space_snapshots(data, outdir, n_cols=4):
    times = data["times"]
    x_mesh = data["x_mesh"]
    v_mesh = data["v_mesh"]
    f_true = data["phase_density_true"]
    f_est = data["phase_density_est"]
    has_nn = "phase_density_nn" in data.files
    f_nn = data["phase_density_nn"] if has_nn else None
    nn_valid = data["phase_density_nn_valid"] if has_nn else None

    n_steps = f_true.shape[0]
    col_idx = np.linspace(0, n_steps - 1, n_cols, dtype=int)

    rows = 3 if has_nn else 2
    fig, axes = plt.subplots(rows, n_cols, figsize=(4 * n_cols, 3.2 * rows))
    if n_cols == 1:
        axes = axes.reshape(rows, 1)

    for c, t in enumerate(col_idx):
        vmax = np.nanmax(f_true[t]) if np.isfinite(f_true[t]).any() else None
        _imshow_density(
            axes[0, c], f_true[t], x_mesh, v_mesh, f"truth  t={times[t]:.2f}", 0, vmax
        )
        _imshow_density(
            axes[1, c], f_est[t], x_mesh, v_mesh, f"estimate  t={times[t]:.2f}", 0, vmax
        )
        if has_nn:
            if nn_valid[t]:
                _imshow_density(
                    axes[2, c],
                    f_nn[t],
                    x_mesh,
                    v_mesh,
                    f"NN target  t={times[t]:.2f}",
                )
            else:
                axes[2, c].text(0.5, 0.5, "no NN target\n(window incomplete)",
                                ha="center", va="center", fontsize=8)
                axes[2, c].set_axis_off()

    fig.suptitle("Phase-space density: truth vs estimate" + (" vs NN target" if has_nn else ""))
    path = os.path.join(outdir, "phase_space_snapshots.png")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _mode1_amplitude(electric, length):
    # |first non-zero Fourier mode| of E along x, per timestep.
    fft = np.fft.rfft(electric, axis=1)
    return np.abs(fft[:, 1]) * (2.0 / electric.shape[1])


def plot_field_diagnostics(data, outdir):
    times = data["times"]
    config = data["config"].item() if hasattr(data["config"], "item") else data["config"]
    length = float(config["length"])
    e_est = data["electric_est"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, _mode1_amplitude(e_est, length), label="estimate", lw=1.5)
    if "electric_truth" in data.files:
        e_truth = data["electric_truth"]
        ax.plot(times, _mode1_amplitude(e_truth, length), label="truth", lw=1.5, alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel("|E| mode-1 amplitude")
    ax.set_yscale("log")
    ax.set_title("Electric-field mode-1 amplitude vs time")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    path = os.path.join(outdir, "field_diagnostics.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot DA nudging results.")
    parser.add_argument("result", help="Path to phase_da_*.npz")
    parser.add_argument("--outdir", default=None, help="Output dir (default: <result_dir>/plots)")
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(os.path.dirname(os.path.abspath(args.result)), "plots")
    os.makedirs(outdir, exist_ok=True)

    data = load_result(args.result)
    print(f"Loaded {args.result}; keys: {sorted(data.files)}")

    paths = [
        plot_error_curves(data, outdir),
        plot_phase_space_snapshots(data, outdir),
        plot_field_diagnostics(data, outdir),
    ]
    print("Wrote:")
    for p in paths:
        print(f"  {p}")
    print(f"Final RMSE to truth: {float(data['phase_err_truth'][-1]):.6f}")
    print(f"Mean  RMSE to truth: {float(np.mean(data['phase_err_truth'])):.6f}")


if __name__ == "__main__":
    main()
