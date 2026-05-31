import argparse
import os

import numpy as np


COMPONENT_NAMES = ("x", "y", "z")
N_COMPONENTS = len(COMPONENT_NAMES)


def load_truth_data(data_path: str):
    """Load truth states with shape (T, N, 3), along with optional times/config."""
    raw = np.load(data_path, allow_pickle=True)
    times = None
    config = {}

    if isinstance(raw, np.ndarray):
        states = raw
    elif isinstance(raw, np.lib.npyio.NpzFile):
        files = set(raw.files)
        if "states" in files:
            states = raw["states"]
        elif {"x_paths", "y_paths", "z_paths"}.issubset(files):
            states = np.stack((raw["x_paths"], raw["y_paths"], raw["z_paths"]), axis=-1)
        elif "arr_0" in files:
            states = raw["arr_0"]
        else:
            raise ValueError(
                f"No 3D state data found in {data_path}. Available keys: {sorted(files)}"
            )

        if "times" in files:
            times = np.asarray(raw["times"], dtype=float)

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
            f"Expected truth states with shape (T, N, 3), got {states.shape}"
        )
    if states.shape[0] < 2:
        raise ValueError(f"Need at least 2 time points, got shape {states.shape}")

    if times is not None:
        if times.ndim != 1 or times.shape[0] != states.shape[0]:
            raise ValueError(
                f"times must have shape ({states.shape[0]},), got {times.shape}"
            )

    return states, times, config


def build_observations(states_true_all, obs_keep_prob, obs_noise_std, seed):
    """
    Generate partial noisy 3D observations.
    mask_missing=1 means a particle is missing at that time for all components.
    """
    if not (0.0 <= obs_keep_prob <= 1.0):
        raise ValueError("obs_keep_prob must be in [0,1]")
    if obs_noise_std < 0.0:
        raise ValueError("obs_noise_std must be nonnegative")

    rng = np.random.default_rng(seed)
    t_len, n_particles, _ = states_true_all.shape

    mask_missing = (rng.random((t_len, n_particles)) > obs_keep_prob).astype(float)
    mask_missing[0] = 0.0

    obs_noise = obs_noise_std * rng.standard_normal(states_true_all.shape)
    states_obs_all = states_true_all + obs_noise
    states_obs_all = np.where(mask_missing[..., None] > 0.5, 0.0, states_obs_all)
    return states_obs_all, mask_missing


def kernel_and_grad_delta(delta, inv_h2, norm_const):
    """
    K_h(delta) = norm_const * exp(-delta^2/(2 h^2)).
    Gradient wrt delta: dK/d(delta) = -(delta/h^2) * K.
    """
    k = norm_const * np.exp(-0.5 * (delta * delta) * inv_h2)
    grad = -delta * inv_h2 * k
    return k, grad


def resolve_lambda_nudge(args):
    lambda_nudge = args.lambda_nudge
    if lambda_nudge is None:
        lambda_nudge = args.mu_nudge
    return float(lambda_nudge)


def resolve_model_params(args, truth_config):
    sigma = args.sigma
    rho = args.rho
    beta = args.beta

    if sigma is None:
        sigma = truth_config.get("sigma", 10.0)
    if rho is None:
        rho = truth_config.get("rho", 28.0)
    if beta is None:
        beta = truth_config.get("beta", 8.0 / 3.0)

    return float(sigma), float(rho), float(beta)


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


def compute_model_drift(states_current, sigma, rho, beta):
    means_now = np.mean(states_current, axis=0)
    drift = np.empty_like(states_current)
    drift[:, 0] = sigma * (states_current[:, 1] - states_current[:, 0])
    drift[:, 1] = states_current[:, 0] * (rho - states_current[:, 2]) - states_current[:, 1]
    drift[:, 2] = states_current[:, 0] * states_current[:, 1] - beta * states_current[:, 2]
    return drift


def build_meshes(states_true_all, mesh_margin, grid_n):
    if grid_n < 1:
        raise ValueError("grid_n must be positive")
    if mesh_margin < 0.0:
        raise ValueError("mesh_margin must be nonnegative")

    mins = np.min(states_true_all, axis=(0, 1)) - mesh_margin
    maxs = np.max(states_true_all, axis=(0, 1)) + mesh_margin
    if np.any(maxs <= mins):
        raise ValueError("Invalid mesh bounds")

    meshes = np.empty((N_COMPONENTS, grid_n), dtype=float)
    dx = np.empty(N_COMPONENTS, dtype=float)
    for comp in range(N_COMPONENTS):
        meshes[comp] = np.linspace(mins[comp], maxs[comp], grid_n, endpoint=False)
        dx[comp] = (maxs[comp] - mins[comp]) / grid_n

    return meshes, dx


def compute_rmse(states_est, states_true):
    sq_error = (states_est - states_true) ** 2
    rmse_total = float(np.sqrt(np.mean(sq_error)))
    rmse_components = np.sqrt(np.mean(sq_error, axis=0))
    return rmse_total, rmse_components


def da_step_kde_nudging_3d(
    states_current,
    states_obs_next,
    mask_next,
    meshes,
    dx,
    dt,
    sigma,
    rho,
    beta,
    h_bw,
    lambda_nudge,
    nudge_iters,
    rng,
):
    """One DA step: 3D Lorenz mean-field forecast plus componentwise KDE nudging."""
    if h_bw <= 0.0:
        raise ValueError("h_bw must be positive")
    if nudge_iters < 1:
        raise ValueError("nudge_iters must be positive")

    n_particles = states_current.shape[0]
    drift = compute_model_drift(states_current, sigma=sigma, rho=rho, beta=beta)
    states_pred = (
        states_current
        + dt * drift
        + np.sqrt(dt) * rng.standard_normal(states_current.shape)
    )

    observed = 1.0 - mask_next
    n_obs = np.sum(observed)
    if n_obs < 1.0:
        nan_components = np.full(N_COMPONENTS, np.nan)
        return states_pred, np.nan, nan_components

    inv_h2 = 1.0 / (h_bw * h_bw)
    norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * h_bw)

    obs_densities = np.empty_like(meshes)
    for comp in range(N_COMPONENTS):
        delta_obs = meshes[comp][:, None] - states_obs_next[None, :, comp]
        k_obs, _ = kernel_and_grad_delta(delta_obs, inv_h2, norm_const)
        k_obs = k_obs * observed[None, :]
        obs_densities[comp] = np.sum(k_obs, axis=1) / n_obs

    residuals = np.zeros_like(meshes)
    for _ in range(nudge_iters):
        control = np.zeros_like(states_pred)
        for comp in range(N_COMPONENTS):
            delta_pred = meshes[comp][:, None] - states_pred[None, :, comp]
            k_pred, grad_pred = kernel_and_grad_delta(delta_pred, inv_h2, norm_const)
            y_hat = np.sum(k_pred, axis=1)/(n_particles)
            residuals[comp] = y_hat - obs_densities[comp]
            control[:, comp] = (lambda_nudge) * dx[comp] * np.einsum(
                "mn,m->n", grad_pred, residuals[comp]
            )
        states_pred = states_pred + (dt / nudge_iters) * control

    mesh_err_components = np.sqrt(np.mean(residuals * residuals, axis=1))
    mesh_err_total = float(np.sqrt(np.mean(residuals * residuals)))
    return states_pred, mesh_err_total, mesh_err_components


def run_data_assimilation(args):
    lambda_nudge = resolve_lambda_nudge(args)
    states_true_all, truth_times, truth_config = load_truth_data(args.data_path)
    t_len, n_particles, _ = states_true_all.shape

    if args.max_steps > 0:
        n_steps = min(args.max_steps, t_len - 1)
    else:
        n_steps = t_len - 1
    if n_steps < 1:
        raise ValueError("Not enough time steps for DA")

    dt, times = resolve_dt_and_times(args, truth_times, n_steps)
    sigma, rho, beta = resolve_model_params(args, truth_config)

    states_obs_all, mask_all = build_observations(
        states_true_all=states_true_all,
        obs_keep_prob=args.obs_keep_prob,
        obs_noise_std=args.obs_noise_std,
        seed=args.seed + 17,
    )
    meshes, dx = build_meshes(states_true_all, args.mesh_margin, args.grid_n)

    rng = np.random.default_rng(args.seed + args.forecast_seed_offset)
    if args.init_from_truth:
        states_est = states_true_all[0].copy()
    else:
        states_est = states_obs_all[0].copy()

    states_true = states_true_all[: n_steps + 1]
    states_obs = states_obs_all[: n_steps + 1]
    mask_missing = mask_all[: n_steps + 1]

    states_est_all = np.empty((n_steps + 1, n_particles, N_COMPONENTS), dtype=float)
    mesh_err = np.empty(n_steps + 1, dtype=float)
    mesh_err_components = np.empty((n_steps + 1, N_COMPONENTS), dtype=float)
    rmse = np.empty(n_steps + 1, dtype=float)
    rmse_components = np.empty((n_steps + 1, N_COMPONENTS), dtype=float)
    means_true = np.mean(states_true, axis=1)
    means_est = np.empty((n_steps + 1, N_COMPONENTS), dtype=float)

    states_est_all[0] = states_est
    mesh_err[0] = 0.0
    mesh_err_components[0] = 0.0
    rmse[0], rmse_components[0] = compute_rmse(states_est, states_true[0])
    means_est[0] = np.mean(states_est, axis=0)

    print(
        f"Running 3D DA: steps={n_steps}, N={n_particles}, dt={dt}, "
        f"lambda={lambda_nudge}, h={args.h_bw}, sigma={sigma}, rho={rho}, beta={beta}, "
        f"keep_prob={args.obs_keep_prob}, obs_noise={args.obs_noise_std}, "
        f"init_from_truth={args.init_from_truth}, forecast_seed_offset={args.forecast_seed_offset}"
    )

    for t in range(n_steps):
        states_obs_next = states_obs[t + 1]
        mask_next = mask_missing[t + 1]

        states_est, err_total, err_components = da_step_kde_nudging_3d(
            states_current=states_est,
            states_obs_next=states_obs_next,
            mask_next=mask_next,
            meshes=meshes,
            dx=dx,
            dt=dt,
            sigma=sigma,
            rho=rho,
            beta=beta,
            h_bw=args.h_bw,
            lambda_nudge=lambda_nudge,
            nudge_iters=args.nudge_iters,
            rng=rng,
        )

        states_est_all[t + 1] = states_est
        mesh_err[t + 1] = err_total
        mesh_err_components[t + 1] = err_components
        rmse[t + 1], rmse_components[t + 1] = compute_rmse(states_est, states_true[t + 1])
        means_est[t + 1] = np.mean(states_est, axis=0)

        if (t + 1) % 50 == 0 or t + 1 == n_steps:
            obs_ratio = float(np.mean(1.0 - mask_next))
            rmse_xyz = ", ".join(
                f"{name}={rmse_components[t + 1, comp]:.6f}"
                for comp, name in enumerate(COMPONENT_NAMES)
            )
            print(
                f"Step {t + 1:5d}/{n_steps} | mesh_err={mesh_err[t + 1]:.6f} "
                f"| rmse={rmse[t + 1]:.6f} | rmse_xyz=({rmse_xyz}) "
                f"| obs_ratio={obs_ratio:.3f}"
            )

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"xyz_da_kdenudge_3d_seed{args.seed}.npz")

    np.savez(
        save_path,
        times=times,
        states_true=states_true,
        states_obs=states_obs,
        mask_missing=mask_missing,
        states_est=states_est_all,
        means_true=means_true,
        means_est=means_est,
        rmse=rmse,
        rmse_components=rmse_components,
        mesh_err=mesh_err,
        mesh_err_components=mesh_err_components,
        V=0.5 * rmse * rmse,
        mesh=meshes,
        mesh_dx=dx,
        x_true=states_true[..., 0],
        y_true=states_true[..., 1],
        z_true=states_true[..., 2],
        x_obs=states_obs[..., 0],
        y_obs=states_obs[..., 1],
        z_obs=states_obs[..., 2],
        x_est=states_est_all[..., 0],
        y_est=states_est_all[..., 1],
        z_est=states_est_all[..., 2],
        mx_true=means_true[:, 0],
        my_true=means_true[:, 1],
        mz_true=means_true[:, 2],
        mx_est=means_est[:, 0],
        my_est=means_est[:, 1],
        mz_est=means_est[:, 2],
        config={
            **vars(args),
            "dt": dt,
            "sigma": sigma,
            "rho": rho,
            "beta": beta,
            "lambda_nudge": lambda_nudge,
        },
    )

    print(f"Saved DA result to: {save_path}")
    print(
        "Final RMSEs: "
        f"total={rmse[-1]:.6f}, "
        f"x={rmse_components[-1, 0]:.6f}, "
        f"y={rmse_components[-1, 1]:.6f}, "
        f"z={rmse_components[-1, 2]:.6f}"
    )
    print(
        "Mean RMSEs : "
        f"total={np.mean(rmse):.6f}, "
        f"x={np.mean(rmse_components[:, 0]):.6f}, "
        f"y={np.mean(rmse_components[:, 1]):.6f}, "
        f"z={np.mean(rmse_components[:, 2]):.6f}"
    )


def parse_args():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.normpath(
        os.path.join(this_dir, "../simulation/data/mv_sim_seed0.npz")
    )
    default_outdir = os.path.join(this_dir, "data")

    parser = argparse.ArgumentParser(
        description="3D data assimilation for the Lorenz-type McKean-Vlasov particle system."
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
        "--sigma",
        type=float,
        default=None,
        help="Lorenz sigma coefficient and diffusion strength. If omitted, reuse the truth config when available.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Lorenz rho parameter. If omitted, reuse the truth config when available.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Lorenz beta parameter. If omitted, reuse the truth config when available.",
    )

    parser.add_argument("--obs_keep_prob", type=float, default=0.3)
    parser.add_argument("--obs_noise_std", type=float, default=0.05)

    parser.add_argument("--grid_n", type=int, default=256)
    parser.add_argument("--h_bw", type=float, default=0.5)
    parser.add_argument(
        "--mu_nudge",
        type=float,
        default=1e-4,
        help="Backward-compatible alias for lambda_nudge.",
    )
    parser.add_argument(
        "--lambda_nudge",
        type=float,
        default=None,
        help="Nudging intensity lambda. If omitted, mu_nudge is used.",
    )
    parser.add_argument("--nudge_iters", type=int, default=20)
    parser.add_argument("--mesh_margin", type=float, default=2.0)
    parser.add_argument(
        "--forecast_seed_offset",
        type=int,
        default=101,
        help="Offset added to seed for forecast noise to avoid artificial correlation with truth generation.",
    )
    parser.add_argument(
        "--init_from_truth",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If true, initialize the DA trajectory from the exact truth state at t=0.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_data_assimilation(args)


if __name__ == "__main__":
    main()
