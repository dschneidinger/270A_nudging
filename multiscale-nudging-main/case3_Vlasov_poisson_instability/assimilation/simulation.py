import argparse
import os
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
SIMULATION_DIR = THIS_DIR.parent / "simulation"
if str(SIMULATION_DIR) not in sys.path:
    sys.path.insert(0, str(SIMULATION_DIR))

# Repo root holds models.py (the f-from-phi architectures). Added so the
# neural-network nudging target can be reconstructed at DA time.
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pic_two_stream_instability import (  # noqa: E402
    SimulationConfig,
    cic_deposit_density,
    cic_gather_field,
    load_particles,
    solve_field_periodic,
)


COMPONENT_NAMES = ("x", "v")
N_COMPONENTS = len(COMPONENT_NAMES)


def phi_from_E(electric, x_grid):
    """Electric potential from the field by cumulative integration in x.

    Mirrors f_from_phi.phi_from_E so the network sees inputs built exactly the
    way it was trained.
    """
    return np.cumsum(-electric * np.gradient(x_grid), axis=1)


def load_phi_to_f_model(exp_dir, n_x, n_v, device):
    """Rebuild a trained phi->f network from an experiment directory.

    `exp_dir` must contain best_model.pth; config.json is looked up there and,
    failing that, in the parent directory (the layout sweep runs produce, where
    config.json sits beside the per-seed run_*/ subdirs).

    Returns (model, config, input_mode, downsample_factor, n_sparse, norm_stats).
    """
    import json

    import torch

    from models import build_model, get_input_mode

    exp_dir = Path(exp_dir)
    ckpt_path = exp_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best_model.pth in {exp_dir}")

    config_path = exp_dir / "config.json"
    if not config_path.exists():
        config_path = exp_dir.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json in {exp_dir} or its parent; cannot rebuild model."
        )
    with open(config_path) as fh:
        config = json.load(fh)

    model_type = config["model_type"]
    input_mode = get_input_mode(model_type)
    downsample_factor = int(config["downsample_factor"])
    # n_sparse is fixed by how many spatial samples the training downsample kept.
    n_sparse = len(range(0, int(n_x), downsample_factor))

    model = build_model(model_type, n_sparse, int(n_x), int(n_v), config)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    norm_stats = ckpt.get("norm_stats")
    if norm_stats is None:
        raise ValueError(
            f"Checkpoint {ckpt_path} has no norm_stats; cannot un-normalize f."
        )

    return model, config, input_mode, downsample_factor, n_sparse, norm_stats


def reconstruct_f_sequence(
    model,
    input_mode,
    norm_stats,
    electric,
    x_grid,
    n_steps,
    downsample_factor,
    n_timesteps,
    target_offset,
    dt,
    device,
    n_x,
    n_v,
    phi_noise_std=0.0,
):
    """Reconstruct grid phase-space density f[t] from sparse truth phi.

    Replicates the exact input pipeline of f_from_phi.prepare_training_data /
    VlasovDataset (sparse phi window, optional residual channel, per-channel
    normalization, dt normalization) so a checkpoint trained there can be
    evaluated here. Returns (f_nn, valid) where f_nn has shape
    (n_steps+1, n_x, n_v) (un-normalized) and `valid[t]` flags timesteps whose
    input window was fully available.
    """
    import torch

    phi = phi_from_E(np.asarray(electric, dtype=float), np.asarray(x_grid, dtype=float))
    phi_sparse = phi[:, ::downsample_factor]  # (N_t, n_sparse)
    if phi_noise_std > 0.0:
        rng = np.random.default_rng(123)
        phi_sparse = phi_sparse + phi_noise_std * rng.standard_normal(phi_sparse.shape)
    n_sparse = phi_sparse.shape[1]

    phi_mean = np.asarray(norm_stats["phi_mean"], dtype=float)  # (n_timesteps, n_sparse)
    phi_std = np.asarray(norm_stats["phi_std"], dtype=float)
    dt_mean = float(norm_stats["dt_mean"])
    dt_std = float(norm_stats["dt_std"])
    f_mean = float(norm_stats["f_mean"])
    f_std = float(norm_stats["f_std"])
    dt_norm = (float(dt) - dt_mean) / dt_std

    f_nn = np.full((n_steps + 1, n_x, n_v), np.nan, dtype=float)
    valid = np.zeros(n_steps + 1, dtype=bool)

    for target_t in range(n_steps + 1):
        window_end = target_t - target_offset
        win_start = window_end - n_timesteps + 1
        if win_start < 0 or window_end >= phi_sparse.shape[0]:
            continue
        window = phi_sparse[win_start : window_end + 1].astype(float)  # (n_timesteps, n_sparse)

        if input_mode == "flat_residual":
            # Second channel carries dphi = phi_t - phi_{t-1} (pre-normalization),
            # matching prepare_training_data.
            window = window.copy()
            window[-2] = phi_sparse[window_end] - phi_sparse[window_end - 1]

        window_norm = (window - phi_mean) / phi_std

        if input_mode in ("flat_pair", "flat_residual"):
            inp = np.concatenate([window_norm[-1], window_norm[-2], [dt_norm]])
        elif input_mode == "sequence":
            dt_row = np.full((1, n_sparse), dt_norm, dtype=float)
            inp = np.concatenate([window_norm, dt_row], axis=0)
        else:
            raise ValueError(f"Unknown input_mode '{input_mode}'")

        x_tensor = torch.as_tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model(x_tensor).cpu().numpy().reshape(n_x, n_v)
        f_nn[target_t] = out * f_std + f_mean
        valid[target_t] = True

    return f_nn, valid


def build_phase_mesh_from_grids(x_grid, v_grid, length):
    """Phase mesh that matches the network's native (x_grid, v_grid).

    Returns the same tuple shape as build_phase_mesh so the rest of the DA code
    is unchanged. x is assumed uniform on [0, length); v uniform from v_grid.
    """
    x_mesh = np.asarray(x_grid, dtype=float)
    v_mesh = np.asarray(v_grid, dtype=float)
    if x_mesh.ndim != 1 or v_mesh.ndim != 1 or x_mesh.size < 4 or v_mesh.size < 4:
        raise ValueError("x_grid and v_grid must be 1D with at least 4 points")
    dx = length / x_mesh.size
    dv = float(v_mesh[1] - v_mesh[0])
    v_min = float(v_mesh[0])
    v_max = float(v_mesh[-1] + dv)
    return x_mesh, v_mesh, dx, dv, v_min, v_max


def load_truth_data(data_path: str):
    """Load truth states together with optional grids and config."""
    raw = np.load(data_path, allow_pickle=True)
    times = None
    config = {}
    phase_density = None
    rho_x = None
    electric = None
    x_grid = None
    v_grid = None

    if isinstance(raw, np.ndarray):
        states = raw
    elif isinstance(raw, np.lib.npyio.NpzFile):
        files = set(raw.files)
        if "states" in files:
            states = raw["states"]
        elif {"x_paths", "v_paths"}.issubset(files):
            states = np.stack((raw["x_paths"], raw["v_paths"]), axis=-1)
        elif "arr_0" in files:
            states = raw["arr_0"]
        else:
            raise ValueError(
                f"No 1D-1V state data found in {data_path}. Available keys: {sorted(files)}"
            )

        if "times" in files:
            times = np.asarray(raw["times"], dtype=float)
        if "phase_density" in files:
            phase_density = np.asarray(raw["phase_density"], dtype=float)
        if "rho_x" in files:
            rho_x = np.asarray(raw["rho_x"], dtype=float)
        if "electric" in files:
            electric = np.asarray(raw["electric"], dtype=float)
        if "x_grid" in files:
            x_grid = np.asarray(raw["x_grid"], dtype=float)
        if "v_grid" in files:
            v_grid = np.asarray(raw["v_grid"], dtype=float)
        if "config" in files:
            config_obj = raw["config"]
            if isinstance(config_obj, np.ndarray) and config_obj.shape == ():
                config_obj = config_obj.item()
            if isinstance(config_obj, dict):
                config = dict(config_obj)
        raw.close()
    else:
        raise ValueError(f"Unsupported data format in {data_path}")

    states = np.asarray(states, dtype=float)
    if states.ndim != 3 or states.shape[-1] != N_COMPONENTS:
        raise ValueError(
            f"Expected truth states with shape (T, N, 2), got {states.shape}"
        )
    if states.shape[0] < 2:
        raise ValueError(f"Need at least 2 time points, got shape {states.shape}")

    if times is not None:
        if times.ndim != 1 or times.shape[0] != states.shape[0]:
            raise ValueError(
                f"times must have shape ({states.shape[0]},), got {times.shape}"
            )

    return states, times, phase_density, rho_x, electric, x_grid, v_grid, config


def sample_particle_indices(n_available, n_target, rng):
    if n_target <= 0:
        raise ValueError("target particle count must be positive")
    if n_target == n_available:
        return np.arange(n_available, dtype=int)
    if n_target < n_available:
        return np.sort(rng.choice(n_available, size=n_target, replace=False))
    return rng.integers(0, n_available, size=n_target, dtype=int)


def subsample_trajectory(states_all, target_n, seed):
    """Select a fixed subset/bootstrap sample of particle trajectories."""
    n_available = states_all.shape[1]
    if target_n is None:
        target_n = n_available

    rng = np.random.default_rng(seed)
    indices = sample_particle_indices(n_available, int(target_n), rng)
    return states_all[:, indices, :].copy(), indices


def resample_state_cloud(states, target_n, seed):
    """Match a single particle cloud to a target particle count."""
    n_available = states.shape[0]
    if target_n is None:
        target_n = n_available

    rng = np.random.default_rng(seed)
    indices = sample_particle_indices(n_available, int(target_n), rng)
    return states[indices].copy()


def wrap_periodic(x, length):
    return np.mod(x, length)


def periodic_delta(delta, length):
    return (delta + 0.5 * length) % length - 0.5 * length


def build_observations(
    states_true_all,
    obs_keep_prob,
    obs_noise_x_std,
    obs_noise_v_std,
    seed,
    length,
):
    """Generate partial noisy observations of (x, v)."""
    if not (0.0 <= obs_keep_prob <= 1.0):
        raise ValueError("obs_keep_prob must be in [0, 1]")
    if obs_noise_x_std < 0.0 or obs_noise_v_std < 0.0:
        raise ValueError("observation noise std must be nonnegative")

    rng = np.random.default_rng(seed)
    t_len, n_particles, _ = states_true_all.shape

    mask_missing = (rng.random((t_len, n_particles)) > obs_keep_prob).astype(float)
    mask_missing[0] = 0.0

    states_obs_all = states_true_all.copy()
    states_obs_all[..., 0] = wrap_periodic(
        states_true_all[..., 0] + obs_noise_x_std * rng.standard_normal((t_len, n_particles)),
        length,
    )
    states_obs_all[..., 1] = (
        states_true_all[..., 1] + obs_noise_v_std * rng.standard_normal((t_len, n_particles))
    )
    states_obs_all = np.where(mask_missing[..., None] > 0.5, 0.0, states_obs_all)
    return states_obs_all, mask_missing


def kernel_and_grad_delta(delta, inv_h2, norm_const):
    k = norm_const * np.exp(-0.5 * (delta * delta) * inv_h2)
    grad = -delta * inv_h2 * k
    return k, grad


def resolve_lambda_nudge(args):
    lambda_nudge = args.lambda_nudge
    if lambda_nudge is None:
        lambda_nudge = args.mu_nudge
    return float(lambda_nudge)


def resolve_dt_and_times(args, truth_times, n_steps):
    if truth_times is not None:
        dt_values = np.diff(truth_times)
        if dt_values.size < 1:
            raise ValueError("Truth trajectory must contain at least two time points")
        dt_data = float(dt_values[0])
        if not np.allclose(dt_values, dt_data):
            raise ValueError("Truth times must be uniformly spaced")
        if args.dt is not None and not np.isclose(args.dt, dt_data):
            raise ValueError(
                f"Provided dt={args.dt} does not match truth data dt={dt_data}"
            )
        return dt_data, np.asarray(truth_times[: n_steps + 1], dtype=float)

    if args.dt is None:
        raise ValueError("Need --dt when the truth file does not contain times")

    dt = float(args.dt)
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return dt, np.linspace(0.0, n_steps * dt, n_steps + 1)


def resolve_model_config(args, truth_config):
    length = float(truth_config.get("length", 4.0 * np.pi))
    vmax_init = float(truth_config.get("vmax", 6.0))
    mode = int(truth_config.get("mode", 1))
    u0 = float(truth_config.get("u0", 1.0))
    sigma = float(truth_config.get("sigma", 0.2))
    random_load = bool(truth_config.get("random_load", False))

    field_grid_n = args.field_grid_n
    if field_grid_n is None:
        field_grid_n = int(truth_config.get("num_grid", 128))
    if field_grid_n < 4:
        raise ValueError("field_grid_n must be at least 4")

    return length, vmax_init, mode, u0, sigma, random_load, int(field_grid_n)


def resolve_truth_initial_profile(truth_config):
    truth_temperature = float(
        truth_config.get("temperature", truth_config.get("sigma", 0.2) ** 2)
    )
    truth_epsilon = float(truth_config.get("epsilon", 0.01))
    truth_phase_shift = float(truth_config.get("phase_shift", 0.0))
    return truth_temperature, truth_epsilon, truth_phase_shift


def resolve_model_initial_profile(args, truth_config):
    truth_temperature, truth_epsilon, truth_phase_shift = resolve_truth_initial_profile(
        truth_config
    )

    if args.init_mode in {"truth", "observed"}:
        init_temperature = truth_temperature
        init_epsilon = truth_epsilon
        init_phase_shift = truth_phase_shift
    elif args.init_mode == "two_stream":
        if args.model_temperature is not None:
            raise ValueError(
                "init_mode='two_stream' does not use model_temperature; "
                "use a Maxwellian init_mode for temperature mismatch studies"
            )
        if args.model_phase_shift is not None:
            raise ValueError(
                "init_mode='two_stream' does not support model_phase_shift; "
                "use perturbed_maxwellian or model_b instead"
            )
        init_temperature = truth_temperature
        init_epsilon = (
            truth_epsilon if args.model_epsilon is None else float(args.model_epsilon)
        )
        init_phase_shift = truth_phase_shift
    elif args.init_mode == "maxwellian":
        init_temperature = (
            truth_temperature
            if args.model_temperature is None
            else float(args.model_temperature)
        )
        init_epsilon = 0.0 if args.model_epsilon is None else float(args.model_epsilon)
        init_phase_shift = (
            0.0 if args.model_phase_shift is None else float(args.model_phase_shift)
        )
    elif args.init_mode == "perturbed_maxwellian":
        init_temperature = (
            truth_temperature
            if args.model_temperature is None
            else float(args.model_temperature)
        )
        init_epsilon = (
            truth_epsilon if args.model_epsilon is None else float(args.model_epsilon)
        )
        init_phase_shift = (
            truth_phase_shift
            if args.model_phase_shift is None
            else float(args.model_phase_shift)
        )
    elif args.init_mode == "model_a":
        init_temperature = (
            1.2 * truth_temperature
            if args.model_temperature is None
            else float(args.model_temperature)
        )
        init_epsilon = (
            truth_epsilon if args.model_epsilon is None else float(args.model_epsilon)
        )
        init_phase_shift = (
            truth_phase_shift
            if args.model_phase_shift is None
            else float(args.model_phase_shift)
        )
    elif args.init_mode == "model_b":
        init_temperature = (
            truth_temperature
            if args.model_temperature is None
            else float(args.model_temperature)
        )
        init_epsilon = (
            truth_epsilon if args.model_epsilon is None else float(args.model_epsilon)
        )
        init_phase_shift = (
            0.25 * np.pi
            if args.model_phase_shift is None
            else float(args.model_phase_shift)
        )
    else:
        raise ValueError(f"Unknown init_mode: {args.init_mode}")

    if init_temperature <= 0.0:
        raise ValueError("model_temperature must be positive")
    if abs(init_epsilon) >= 1.0:
        raise ValueError("model_epsilon must satisfy |model_epsilon| < 1")

    return (
        truth_temperature,
        truth_epsilon,
        truth_phase_shift,
        init_temperature,
        init_epsilon,
        init_phase_shift,
    )


def physical_wavenumber(length, mode):
    return 2.0 * np.pi * mode / length


def quiet_start_positions_with_phase(num_particles, length, epsilon, mode, phase_shift):
    if abs(epsilon) >= 1.0:
        raise ValueError("quiet-start loading requires |epsilon| < 1.")

    k = physical_wavenumber(length, mode)
    u = (np.arange(num_particles, dtype=np.float64) + 0.5) / num_particles
    x = length * u

    if epsilon != 0.0:
        for _ in range(8):
            residual = (
                x / length
                + epsilon
                * (np.sin(k * x + phase_shift) - np.sin(phase_shift))
                / (length * k)
                - u
            )
            derivative = (1.0 + epsilon * np.cos(k * x + phase_shift)) / length
            x -= residual / derivative

    return np.mod(x, length)


def random_positions_with_phase(
    num_particles, length, epsilon, mode, phase_shift, rng
):
    if abs(epsilon) >= 1.0:
        raise ValueError("random loading requires |epsilon| < 1.")

    k = physical_wavenumber(length, mode)
    max_pdf = 1.0 + abs(epsilon)
    accepted = []
    remaining = num_particles

    while remaining > 0:
        batch = max(2 * remaining, 1024)
        x_trial = rng.uniform(0.0, length, size=batch)
        y_trial = rng.uniform(0.0, max_pdf, size=batch)
        keep = y_trial < (1.0 + epsilon * np.cos(k * x_trial + phase_shift))
        accepted_batch = x_trial[keep]
        if accepted_batch.size:
            take = min(remaining, accepted_batch.size)
            accepted.append(accepted_batch[:take])
            remaining -= take

    return np.concatenate(accepted)


def quiet_start_velocities_temperature(num_particles, vmax, temperature, rng):
    sigma = np.sqrt(temperature)
    normal = NormalDist(mu=0.0, sigma=sigma)
    cdf_low = normal.cdf(-vmax)
    cdf_high = normal.cdf(vmax)
    u = (np.arange(num_particles, dtype=np.float64) + 0.5) / num_particles
    q = cdf_low + (cdf_high - cdf_low) * u
    velocities = np.array([normal.inv_cdf(value) for value in q], dtype=np.float64)
    rng.shuffle(velocities)
    return velocities


def random_velocities_temperature(num_particles, vmax, temperature, rng):
    sigma = np.sqrt(temperature)
    accepted = []
    remaining = num_particles

    while remaining > 0:
        batch = max(2 * remaining, 1024)
        v_trial = rng.normal(loc=0.0, scale=sigma, size=batch)
        keep = np.abs(v_trial) <= vmax
        accepted_batch = v_trial[keep]
        if accepted_batch.size:
            take = min(remaining, accepted_batch.size)
            accepted.append(accepted_batch[:take])
            remaining -= take

    return np.concatenate(accepted)


def sample_maxwellian_model_initial_state(
    n_particles,
    length,
    vmax,
    mode,
    epsilon,
    phase_shift,
    temperature,
    random_load,
    seed,
):
    rng = np.random.default_rng(seed)
    if random_load:
        x0 = random_positions_with_phase(
            n_particles, length, epsilon, mode, phase_shift, rng
        )
        v0 = random_velocities_temperature(n_particles, vmax, temperature, rng)
    else:
        x0 = quiet_start_positions_with_phase(
            n_particles, length, epsilon, mode, phase_shift
        )
        v0 = quiet_start_velocities_temperature(n_particles, vmax, temperature, rng)
    return np.column_stack((x0, v0))


def build_phase_mesh(states_true, length, grid_nx, grid_nv, v_mesh_margin):
    if grid_nx < 4 or grid_nv < 4:
        raise ValueError("grid_nx and grid_nv must both be at least 4")
    if v_mesh_margin < 0.0:
        raise ValueError("v_mesh_margin must be nonnegative")

    x_mesh = np.linspace(0.0, length, grid_nx, endpoint=False, dtype=float)
    v_min = float(np.min(states_true[..., 1])) - v_mesh_margin
    v_max = float(np.max(states_true[..., 1])) + v_mesh_margin
    if v_max <= v_min:
        raise ValueError("Invalid velocity mesh bounds")
    v_mesh = np.linspace(v_min, v_max, grid_nv, endpoint=False, dtype=float)
    dx = length / grid_nx
    dv = (v_max - v_min) / grid_nv
    return x_mesh, v_mesh, dx, dv, v_min, v_max


def resolve_bandwidths(args, dx, dv):
    hx_bw = args.hx_bw if args.hx_bw is not None else 2.0 * dx
    hv_bw = args.hv_bw if args.hv_bw is not None else 2.0 * dv
    if hx_bw <= 0.0 or hv_bw <= 0.0:
        raise ValueError("hx_bw and hv_bw must be positive")
    return float(hx_bw), float(hv_bw)


def phase_kernels(states, x_mesh, v_mesh, length, hx_bw, hv_bw):
    inv_hx2 = 1.0 / (hx_bw * hx_bw)
    inv_hv2 = 1.0 / (hv_bw * hv_bw)
    norm_x = 1.0 / (np.sqrt(2.0 * np.pi) * hx_bw)
    norm_v = 1.0 / (np.sqrt(2.0 * np.pi) * hv_bw)

    delta_x = periodic_delta(x_mesh[:, None] - states[None, :, 0], length)
    delta_v = v_mesh[:, None] - states[None, :, 1]

    kx, gx = kernel_and_grad_delta(delta_x, inv_hx2, norm_x)
    kv, gv = kernel_and_grad_delta(delta_v, inv_hv2, norm_v)
    return kx, gx, kv, gv


def density_from_kernels(kx, kv, weights=None):
    if weights is None:
        return np.einsum("in,jn->ij", kx, kv)
    return np.einsum("in,jn,n->ij", kx, kv, weights)


def build_phase_density(states, x_mesh, v_mesh, length, hx_bw, hv_bw, weights=None):
    kx, _, kv, _ = phase_kernels(states, x_mesh, v_mesh, length, hx_bw, hv_bw)
    return density_from_kernels(kx, kv, weights=weights)


def phase_rmse(density_est, density_true):
    return float(np.sqrt(np.mean((density_est - density_true) ** 2)))


def compute_field_diagnostics(states, length, field_grid_n):
    n_particles = states.shape[0]
    particle_weight = length / n_particles
    rho_x = cic_deposit_density(states[:, 0], particle_weight, field_grid_n, length)
    _, electric = solve_field_periodic(rho_x, length)
    return rho_x, electric


def forecast_step_pic(states_current, dt, length, field_grid_n):
    x = states_current[:, 0]
    v = states_current[:, 1]
    rho_x, electric = compute_field_diagnostics(states_current, length, field_grid_n)

    e_particle = cic_gather_field(x, electric, length)
    v_half = v + 0.5 * dt * e_particle
    x_pred = wrap_periodic(x + dt * v_half, length)

    states_mid = np.column_stack((x_pred, v_half))
    rho_x_pred, electric_pred = compute_field_diagnostics(states_mid, length, field_grid_n)
    e_particle = cic_gather_field(x_pred, electric_pred, length)
    v_pred = v_half + 0.5 * dt * e_particle

    states_pred = np.column_stack((x_pred, v_pred))
    return states_pred, rho_x_pred, electric_pred


def initialize_estimate(
    init_mode,
    states_true0,
    states_obs0,
    n_particles,
    length,
    vmax_init,
    mode,
    u0,
    sigma,
    init_temperature,
    init_epsilon,
    init_phase_shift,
    random_load,
    seed,
):
    if init_mode == "truth":
        return resample_state_cloud(states_true0, n_particles, seed)
    if init_mode == "observed":
        return resample_state_cloud(states_obs0, n_particles, seed)
    if init_mode == "two_stream":
        rng = np.random.default_rng(seed)
        init_cfg = SimulationConfig(
            num_particles=n_particles,
            num_grid=8,
            num_v_grid=8,
            length=length,
            t_end=1.0,
            dt=1.0,
            epsilon=init_epsilon,
            mode=mode,
            vmax=vmax_init,
            u0=u0,
            sigma=sigma,
            seed=seed,
            output_dir="unused",
            random_load=random_load,
        )
        x0, v0 = load_particles(init_cfg, rng)
        return np.column_stack((x0, v0))
    if init_mode not in {"maxwellian", "perturbed_maxwellian", "model_a", "model_b"}:
        raise ValueError(f"Unknown init_mode: {init_mode}")
    return sample_maxwellian_model_initial_state(
        n_particles=n_particles,
        length=length,
        vmax=vmax_init,
        mode=mode,
        epsilon=init_epsilon,
        phase_shift=init_phase_shift,
        temperature=init_temperature,
        random_load=random_load,
        seed=seed,
    )


def da_step_kde_nudging_1d1v(
    states_current,
    states_obs_next,
    mask_next,
    x_mesh,
    v_mesh,
    dx,
    dv,
    dt,
    length,
    field_grid_n,
    hx_bw,
    hv_bw,
    lambda_nudge,
    nudge_iters,
    target_density=None,
):
    """One DA step: PIC forecast followed by 2D phase-space density nudging.

    The nudging target is `target_density` when supplied (e.g. a network's f
    reconstructed from sparse phi); otherwise it is a KDE of the particle
    observations, as before.
    """
    if nudge_iters < 1:
        raise ValueError("nudge_iters must be positive")

    states_pred, rho_x_pred, electric_pred = forecast_step_pic(
        states_current, dt, length, field_grid_n
    )
    n_particles = states_pred.shape[0]

    if target_density is not None:
        density_obs = np.asarray(target_density, dtype=float)
    else:
        observed = 1.0 - mask_next
        n_obs = float(np.sum(observed))
        if n_obs < 1.0:
            density_pred = build_phase_density(
                states_pred, x_mesh, v_mesh, length, hx_bw, hv_bw
            )
            return states_pred, np.nan, density_pred, rho_x_pred, electric_pred

        obs_weights = observed / n_obs
        density_obs = build_phase_density(
            states_obs_next,
            x_mesh,
            v_mesh,
            length,
            hx_bw,
            hv_bw,
            weights=obs_weights,
        )

    residual = None
    for _ in range(nudge_iters):
        kx_pred, gx_pred, kv_pred, gv_pred = phase_kernels(
            states_pred, x_mesh, v_mesh, length, hx_bw, hv_bw
        )
        density_pred = density_from_kernels(kx_pred, kv_pred) / n_particles
        residual = density_pred - density_obs

        scale = lambda_nudge * dx * dv
        rx = residual @ kv_pred
        rv = residual.T @ kx_pred
        control_x = scale * np.sum(gx_pred * rx, axis=0)
        control_v = scale * np.sum(gv_pred * rv, axis=0)

        states_pred[:, 0] = wrap_periodic(
            states_pred[:, 0] + (dt / nudge_iters) * control_x,
            length,
        )
        states_pred[:, 1] = states_pred[:, 1] + (dt / nudge_iters) * control_v

    density_pred = build_phase_density(states_pred, x_mesh, v_mesh, length, hx_bw, hv_bw)
    rho_x_pred, electric_pred = compute_field_diagnostics(states_pred, length, field_grid_n)
    mesh_err_total = float(np.sqrt(np.mean(residual * residual)))
    return states_pred, mesh_err_total, density_pred, rho_x_pred, electric_pred


def run_data_assimilation(args):
    lambda_nudge = resolve_lambda_nudge(args)
    (
        states_true_all,
        truth_times,
        phase_density_truth_file,
        rho_x_truth_file,
        electric_truth_file,
        x_grid_file,
        v_grid_file,
        truth_config,
    ) = load_truth_data(args.data_path)
    t_len, n_truth_particles, _ = states_true_all.shape

    if args.max_steps > 0:
        n_steps = min(args.max_steps, t_len - 1)
    else:
        n_steps = t_len - 1
    if n_steps < 1:
        raise ValueError("Not enough time steps for DA")

    dt, times = resolve_dt_and_times(args, truth_times, n_steps)
    (
        length,
        vmax_init,
        mode,
        u0,
        sigma,
        random_load,
        field_grid_n,
    ) = resolve_model_config(args, truth_config)
    (
        truth_temperature,
        truth_epsilon,
        truth_phase_shift,
        init_temperature,
        init_epsilon,
        init_phase_shift,
    ) = resolve_model_initial_profile(args, truth_config)

    if args.num_particles is None:
        n_est_particles = n_truth_particles
    else:
        n_est_particles = int(args.num_particles)
    if n_est_particles <= 0:
        raise ValueError("num_particles must be positive")

    if args.obs_num_particles is None:
        n_obs_particles = n_est_particles
    else:
        n_obs_particles = int(args.obs_num_particles)
    if n_obs_particles <= 0:
        raise ValueError("obs_num_particles must be positive")

    states_true = states_true_all[: n_steps + 1]

    # When a phi->f network drives the nudging, the target f lives on the
    # network's native (x_grid, v_grid). Align the DA phase mesh to it so the
    # reconstructed density can be used directly without interpolation.
    nn_mode = args.phi_to_f_model is not None
    if nn_mode:
        if x_grid_file is None or v_grid_file is None or electric_truth_file is None:
            raise ValueError(
                "--phi_to_f_model requires the truth file to contain x_grid, "
                "v_grid, and electric."
            )
        x_mesh, v_mesh, dx, dv, v_min, v_max = build_phase_mesh_from_grids(
            x_grid_file, v_grid_file, length
        )
        grid_nx, grid_nv = x_mesh.size, v_mesh.size
    else:
        x_mesh, v_mesh, dx, dv, v_min, v_max = build_phase_mesh(
            states_true, length, args.grid_nx, args.grid_nv, args.v_mesh_margin
        )
        grid_nx, grid_nv = args.grid_nx, args.grid_nv
    hx_bw, hv_bw = resolve_bandwidths(args, dx, dv)

    # Reconstruct the network's f-target sequence once, up front. Each target is
    # clipped to be nonnegative and renormalized to unit phase-space mass so it
    # matches the per-particle-normalized density the nudge compares against.
    f_nn = None
    f_nn_valid = None
    nn_model_meta = None
    if nn_mode:
        import torch

        nn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, model_cfg, input_mode, ds_factor, n_sparse_m, norm_stats = (
            load_phi_to_f_model(args.phi_to_f_model, grid_nx, grid_nv, nn_device)
        )
        nn_model_meta = {
            "model_type": model_cfg["model_type"],
            "input_mode": input_mode,
            "downsample_factor": ds_factor,
            "n_timesteps": int(model_cfg["n_timesteps"]),
            "target_offset": int(model_cfg.get("target_offset", 0)),
            "n_sparse": n_sparse_m,
        }
        print(
            f"phi->f nudging target: {args.phi_to_f_model} "
            f"(model={nn_model_meta['model_type']}, input_mode={input_mode}, "
            f"ds={ds_factor}, n_sparse={n_sparse_m}, "
            f"n_timesteps={nn_model_meta['n_timesteps']}, "
            f"target_offset={nn_model_meta['target_offset']})"
        )
        f_nn, f_nn_valid = reconstruct_f_sequence(
            model=model,
            input_mode=input_mode,
            norm_stats=norm_stats,
            electric=electric_truth_file[: n_steps + 1],
            x_grid=x_mesh,
            n_steps=n_steps,
            downsample_factor=ds_factor,
            n_timesteps=nn_model_meta["n_timesteps"],
            target_offset=nn_model_meta["target_offset"],
            dt=dt,
            device=nn_device,
            n_x=grid_nx,
            n_v=grid_nv,
            phi_noise_std=args.phi_noise_std,
        )
        for t in range(n_steps + 1):
            if f_nn_valid[t]:
                f_pos = np.clip(f_nn[t], 0.0, None)
                mass = float(f_pos.sum() * dx * dv)
                f_nn[t] = f_pos / mass if mass > 0.0 else f_pos
        n_valid = int(np.sum(f_nn_valid))
        print(
            f"Reconstructed f targets: {n_valid}/{n_steps + 1} timesteps have a "
            f"full phi window (earlier steps fall back to observation nudging)."
        )

    states_obs_source_all, obs_indices = subsample_trajectory(
        states_true_all, n_obs_particles, seed=args.seed + 29
    )
    states_obs_all, mask_all = build_observations(
        states_true_all=states_obs_source_all,
        obs_keep_prob=args.obs_keep_prob,
        obs_noise_x_std=args.obs_noise_x_std,
        obs_noise_v_std=args.obs_noise_v_std,
        seed=args.seed + 17,
        length=length,
    )
    states_obs = states_obs_all[: n_steps + 1]
    mask_missing = mask_all[: n_steps + 1]

    phase_density_true = np.empty((n_steps + 1, grid_nx, grid_nv), dtype=float)
    phase_density_obs = np.empty_like(phase_density_true)
    truth_weights = np.full(
        n_truth_particles, n_est_particles / n_truth_particles, dtype=float
    )
    for t in range(n_steps + 1):
        phase_density_true[t] = build_phase_density(
            states_true[t],
            x_mesh,
            v_mesh,
            length,
            hx_bw,
            hv_bw,
            weights=truth_weights,
        )
        observed = 1.0 - mask_missing[t]
        n_obs = float(np.sum(observed))
        if n_obs < 1.0:
            phase_density_obs[t] = np.nan
        else:
            obs_weights = observed * (n_est_particles / n_obs)
            phase_density_obs[t] = build_phase_density(
                states_obs[t], x_mesh, v_mesh, length, hx_bw, hv_bw, weights=obs_weights
            )

    states_est = initialize_estimate(
        init_mode=args.init_mode,
        states_true0=states_true[0],
        states_obs0=states_obs[0],
        n_particles=n_est_particles,
        length=length,
        vmax_init=vmax_init,
        mode=mode,
        u0=u0,
        sigma=sigma,
        init_temperature=init_temperature,
        init_epsilon=init_epsilon,
        init_phase_shift=init_phase_shift,
        random_load=random_load,
        seed=args.seed + args.init_seed_offset,
    )

    states_est_all = np.empty((n_steps + 1, n_est_particles, N_COMPONENTS), dtype=float)
    phase_density_est = np.empty_like(phase_density_true)
    phase_err_obs = np.empty(n_steps + 1, dtype=float)
    phase_err_truth = np.empty(n_steps + 1, dtype=float)
    rho_x_est = np.empty((n_steps + 1, field_grid_n), dtype=float)
    electric_est = np.empty((n_steps + 1, field_grid_n), dtype=float)

    states_est_all[0] = states_est
    phase_density_est[0] = build_phase_density(
        states_est, x_mesh, v_mesh, length, hx_bw, hv_bw
    )
    phase_err_obs[0] = phase_rmse(phase_density_est[0], phase_density_obs[0])
    phase_err_truth[0] = phase_rmse(phase_density_est[0], phase_density_true[0])
    rho_x_est[0], electric_est[0] = compute_field_diagnostics(
        states_est, length, field_grid_n
    )

    print(
        f"Running 1D-1V DA: steps={n_steps}, N_truth={n_truth_particles}, "
        f"N_obs={n_obs_particles}, N_est={n_est_particles}, dt={dt}, "
        f"lambda={lambda_nudge}, hx={hx_bw}, hv={hv_bw}, "
        f"init_mode={args.init_mode}, init_T={init_temperature}, "
        f"init_eps={init_epsilon}, init_theta={init_phase_shift}, "
        f"keep_prob={args.obs_keep_prob}, "
        f"obs_noise_x={args.obs_noise_x_std}, obs_noise_v={args.obs_noise_v_std}"
    )

    for t in range(n_steps):
        # In NN mode, nudge toward the reconstructed f when its phi window is
        # available; otherwise (and in observation mode) fall back to the
        # observation-based KDE target.
        target_density = None
        if nn_mode and f_nn_valid[t + 1]:
            target_density = f_nn[t + 1]

        states_est, err_obs, density_est, rho_x_now, electric_now = da_step_kde_nudging_1d1v(
            states_current=states_est,
            states_obs_next=states_obs[t + 1],
            mask_next=mask_missing[t + 1],
            x_mesh=x_mesh,
            v_mesh=v_mesh,
            dx=dx,
            dv=dv,
            dt=dt,
            length=length,
            field_grid_n=field_grid_n,
            hx_bw=hx_bw,
            hv_bw=hv_bw,
            lambda_nudge=lambda_nudge,
            nudge_iters=args.nudge_iters,
            target_density=target_density,
        )

        states_est_all[t + 1] = states_est
        phase_density_est[t + 1] = density_est
        phase_err_obs[t + 1] = err_obs
        phase_err_truth[t + 1] = phase_rmse(density_est, phase_density_true[t + 1])
        rho_x_est[t + 1] = rho_x_now
        electric_est[t + 1] = electric_now

        if (t + 1) % 50 == 0 or t + 1 == n_steps:
            obs_ratio = float(np.mean(1.0 - mask_missing[t + 1]))
            print(
                f"Step {t + 1:5d}/{n_steps} | phase_err_obs={phase_err_obs[t + 1]:.6f} "
                f"| phase_err_truth={phase_err_truth[t + 1]:.6f} "
                f"| obs_ratio={obs_ratio:.3f}"
            )

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"phase_da_kdenudge_1d1v_seed{args.seed}.npz")

    payload = {
        "times": times,
        "states_true": states_true,
        "states_obs": states_obs,
        "mask_missing": mask_missing,
        "obs_indices": obs_indices,
        "states_est": states_est_all,
        "phase_density_true": phase_density_true,
        "phase_density_obs": phase_density_obs,
        "phase_density_est": phase_density_est,
        "phase_err_obs": phase_err_obs,
        "phase_err_truth": phase_err_truth,
        "rho_x_est": rho_x_est,
        "electric_est": electric_est,
        "x_mesh": x_mesh,
        "v_mesh": v_mesh,
        "x_true": states_true[..., 0],
        "v_true": states_true[..., 1],
        "x_obs": states_obs[..., 0],
        "v_obs": states_obs[..., 1],
        "x_est": states_est_all[..., 0],
        "v_est": states_est_all[..., 1],
        "config": {
            **vars(args),
            "dt": dt,
            "length": length,
            "field_grid_n": field_grid_n,
            "vmax_init": vmax_init,
            "mode": mode,
            "u0": u0,
            "sigma": sigma,
            "truth_temperature": truth_temperature,
            "truth_epsilon": truth_epsilon,
            "truth_phase_shift": truth_phase_shift,
            "init_temperature": init_temperature,
            "init_epsilon": init_epsilon,
            "init_phase_shift": init_phase_shift,
            "n_truth_particles": n_truth_particles,
            "n_obs_particles": n_obs_particles,
            "n_est_particles": n_est_particles,
            "lambda_nudge": lambda_nudge,
            "hx_bw": hx_bw,
            "hv_bw": hv_bw,
            "v_mesh_min": v_min,
            "v_mesh_max": v_max,
        },
    }

    if nn_mode:
        payload["phase_density_nn"] = f_nn
        payload["phase_density_nn_valid"] = f_nn_valid
        payload["config"]["phi_to_f_model"] = args.phi_to_f_model
        payload["config"]["phi_to_f_meta"] = nn_model_meta

    if phase_density_truth_file is not None:
        payload["phase_density_truth_file"] = phase_density_truth_file[: n_steps + 1]
    if rho_x_truth_file is not None:
        payload["rho_x_truth"] = rho_x_truth_file[: n_steps + 1]
    if electric_truth_file is not None:
        payload["electric_truth"] = electric_truth_file[: n_steps + 1]

    np.savez(save_path, **payload)

    print(f"Saved DA result to: {save_path}")
    print(f"Final phase-space RMSE to truth: {phase_err_truth[-1]:.6f}")
    print(f"Mean  phase-space RMSE to truth: {np.mean(phase_err_truth):.6f}")


def parse_args():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.normpath(
        os.path.join(this_dir, "../simulation/data/mv_sim_seed0.npz")
    )
    default_outdir = os.path.join(this_dir, "data")

    parser = argparse.ArgumentParser(
        description=(
            "1D-1V data assimilation for Vlasov-Poisson using phase-space KDE nudging."
        )
    )

    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--outdir", type=str, default=default_outdir)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step. If omitted and the truth file stores times, reuse that dt.",
    )
    parser.add_argument(
        "--num_particles",
        type=int,
        default=1000,
        help="Number of particles used by the DA forecast / estimator system.",
    )
    parser.add_argument(
        "--obs_num_particles",
        type=int,
        default=None,
        help="Number of truth particles subsampled into the DA observations. Defaults to num_particles.",
    )
    parser.add_argument("--field_grid_n", type=int, default=None)

    parser.add_argument("--obs_keep_prob", type=float, default=1.0)
    parser.add_argument("--obs_noise_x_std", type=float, default=0.0)
    parser.add_argument("--obs_noise_v_std", type=float, default=0.0)

    parser.add_argument(
        "--phi_to_f_model",
        type=str,
        default=None,
        help=(
            "Path to a trained f-from-phi experiment directory (containing "
            "best_model.pth and config.json). When set, the nudging target f is "
            "reconstructed by this network from sparse phi measurements of the "
            "truth electric field, replacing the particle-observation target. "
            "The DA phase mesh is aligned to the network's (x_grid, v_grid). "
            "When omitted, behavior is unchanged."
        ),
    )
    parser.add_argument(
        "--phi_noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to sparse phi inputs at inference (0 = clean).",
    )

    parser.add_argument("--grid_nx", type=int, default=64)
    parser.add_argument("--grid_nv", type=int, default=64)
    parser.add_argument("--v_mesh_margin", type=float, default=0.5)
    parser.add_argument("--hx_bw", type=float, default=None)
    parser.add_argument("--hv_bw", type=float, default=None)
    parser.add_argument(
        "--mu_nudge",
        type=float,
        default=1e-3,
        help="Backward-compatible alias for lambda_nudge.",
    )
    parser.add_argument(
        "--lambda_nudge",
        type=float,
        default=None,
        help="Nudging intensity lambda. If omitted, mu_nudge is used.",
    )
    parser.add_argument("--nudge_iters", type=int, default=5)
    parser.add_argument(
        "--init_mode",
        type=str,
        choices=(
            "two_stream",
            "maxwellian",
            "perturbed_maxwellian",
            "model_a",
            "model_b",
            "observed",
            "truth",
        ),
        default="two_stream",
        help="Initial state for the forecast / prediction trajectory.",
    )
    parser.add_argument(
        "--model_temperature",
        type=float,
        default=None,
        help=(
            "Temperature T~ of the model Maxwellian. For this instability case, "
            "the truth bath temperature defaults to sigma^2, and model_a defaults "
            "to 1.2 times that truth bath temperature when omitted."
        ),
    )
    parser.add_argument(
        "--model_epsilon",
        type=float,
        default=None,
        help=(
            "Perturbation amplitude eps~ in 1 + eps~ cos(kx + theta). "
            "Defaults depend on init_mode."
        ),
    )
    parser.add_argument(
        "--model_phase_shift",
        type=float,
        default=None,
        help=(
            "Phase shift theta in cos(kx + theta). model_b defaults to theta=pi/4 "
            "when omitted."
        ),
    )
    parser.add_argument(
        "--init_seed_offset",
        type=int,
        default=101,
        help="Offset added to seed when sampling the model initial ensemble.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_data_assimilation(args)


if __name__ == "__main__":
    main()
