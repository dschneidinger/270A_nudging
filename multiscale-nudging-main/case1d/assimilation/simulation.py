import argparse
import os
import numpy as np


def load_truth_paths(data_path: str):
    """Load truth trajectory and return array with shape (T, N)."""
    raw = np.load(data_path, allow_pickle=True)

    if isinstance(raw, np.ndarray):
        arr = raw
    elif isinstance(raw, np.lib.npyio.NpzFile):
        if "x_paths" in raw.files:
            arr = raw["x_paths"]
        elif "arr_0" in raw.files:
            arr = raw["arr_0"]
        else:
            raise ValueError(f"No x_paths/arr_0 found in {data_path}. Keys: {raw.files}")
    else:
        raise ValueError(f"Unsupported data format in {data_path}")

    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        x_true = arr
    elif arr.ndim == 3:
        x_true = arr[..., 0]
    else:
        raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")

    if isinstance(raw, np.lib.npyio.NpzFile):
        raw.close()

    if x_true.shape[0] < 2:
        raise ValueError(f"Need at least 2 time points, got shape {x_true.shape}")
    return x_true


def build_observations(x_true_all, obs_keep_prob, obs_noise_std, seed):
    """
    Generate partial noisy observations.
    mask_missing=1 means missing, 0 means observed.
    """
    if not (0.0 <= obs_keep_prob <= 1.0):
        raise ValueError("obs_keep_prob must be in [0,1]")

    rng = np.random.default_rng(seed)
    t_len, n_particles = x_true_all.shape

    mask_missing = (rng.random((t_len, n_particles)) > obs_keep_prob).astype(float)
    mask_missing[0] = 0.0

    obs_noise = obs_noise_std * rng.standard_normal((t_len, n_particles))
    x_obs_all = x_true_all + obs_noise
    x_obs_all = np.where(mask_missing > 0.5, 0.0, x_obs_all)
    return x_obs_all, mask_missing


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


def compute_model_drift(x_current, a, c):
    m_now = np.mean(x_current)
    return -a * (x_current -  m_now)


def da_step_kde_nudging_1d(
    x_current,
    x_obs_next,
    mask_next,
    z,
    dx,
    dt,
    a,
    c,
    sigma,
    h_bw,
    lambda_nudge,
    nudge_iters,
    rng,
):
    """One DA step: forecast with MV drift + KDE nudging assimilation."""
    n_particles = x_current.shape[0]

    drift = compute_model_drift(x_current, a=a, c=c)
    x_pred = x_current + dt * drift + sigma * np.sqrt(dt) * rng.standard_normal(n_particles)

    observed = 1.0 - mask_next
    n_obs = np.sum(observed)
    if n_obs < 1.0:
        return x_pred, np.nan

    inv_h2 = 1.0 / (h_bw * h_bw)
    norm_const = 1.0 / (np.sqrt(2.0 * np.pi) * h_bw)

    delta_obs = z[:, None] - x_obs_next[None, :]
    k_obs, _ = kernel_and_grad_delta(delta_obs, inv_h2, norm_const)
    k_obs = k_obs * observed[None, :]
    y_obs = np.sum(k_obs, axis=1) / n_obs

    r = np.zeros_like(y_obs)
    for _ in range(nudge_iters):
        delta_pred = z[:, None] - x_pred[None, :]
        k_pred, grad_pred = kernel_and_grad_delta(delta_pred, inv_h2, norm_const)
        y_hat = np.sum(k_pred, axis=1)/(n_particles)
        r = y_hat - y_obs
        u = lambda_nudge * dx * np.einsum("mn,m->n", grad_pred, r)
        x_pred = x_pred + dt/nudge_iters * u

    rms_mesh_error = float(np.sqrt(np.mean(r * r)))
    return x_pred, rms_mesh_error


def run_data_assimilation(args):
    lambda_nudge = resolve_lambda_nudge(args)
    x_true_all = load_truth_paths(args.data_path)
    t_len, n_particles = x_true_all.shape

    if args.max_steps > 0:
        n_steps = min(args.max_steps, t_len - 1)
    else:
        n_steps = t_len - 1
    if n_steps < 1:
        raise ValueError("Not enough time steps for DA")

    x_obs_all, mask_all = build_observations(
        x_true_all=x_true_all,
        obs_keep_prob=args.obs_keep_prob,
        obs_noise_std=args.obs_noise_std,
        seed=args.seed + 17,
    )

    x_min = float(np.min(x_true_all)) - args.mesh_margin
    x_max = float(np.max(x_true_all)) + args.mesh_margin
    if x_max <= x_min:
        raise ValueError("Invalid mesh bounds")
    z = np.linspace(x_min, x_max, args.grid_n, endpoint=False)
    dx = (x_max - x_min) / args.grid_n

    rng = np.random.default_rng(args.seed + args.forecast_seed_offset)
    if args.init_from_truth:
        x_est = x_true_all[0].copy()
    else:
        x_est = x_obs_all[0].copy()

    x_est_all = np.empty((n_steps + 1, n_particles), dtype=float)
    mesh_err = np.empty(n_steps + 1, dtype=float)
    rmse = np.empty(n_steps + 1, dtype=float)

    x_est_all[0] = x_est
    mesh_err[0] = 0.0
    rmse[0] = float(np.sqrt(np.mean((x_est - x_true_all[0]) ** 2)))

    print(
        f"Running 1D DA: steps={n_steps}, N={n_particles}, dt={args.dt}, "
        f"lambda={lambda_nudge}, h={args.h_bw}, a={args.a}, c={args.c}, "
        f"keep_prob={args.obs_keep_prob}, obs_noise={args.obs_noise_std}, "
        f"init_from_truth={args.init_from_truth}, forecast_seed_offset={args.forecast_seed_offset}"
    )

    times = np.linspace(0.0, n_steps * args.dt, n_steps + 1)
    for t in range(n_steps):
        x_obs_next = x_obs_all[t + 1]
        mask_next = mask_all[t + 1]

        x_est, err = da_step_kde_nudging_1d(
            x_current=x_est,
            x_obs_next=x_obs_next,
            mask_next=mask_next,
            z=z,
            dx=dx,
            dt=args.dt,
            a=args.a,
            c=args.c,
            sigma=args.sigma,
            h_bw=args.h_bw,
            lambda_nudge=lambda_nudge,
            nudge_iters=args.nudge_iters,
            rng=rng,
        )

        x_est_all[t + 1] = x_est
        mesh_err[t + 1] = err
        rmse[t + 1] = float(np.sqrt(np.mean((x_est - x_true_all[t + 1]) ** 2)))

        if (t + 1) % 50 == 0 or t + 1 == n_steps:
            obs_ratio = float(np.mean(1.0 - mask_next))
            print(
                f"Step {t + 1:5d}/{n_steps} | mesh_err={mesh_err[t + 1]:.6f} "
                f"| rmse={rmse[t + 1]:.6f} | obs_ratio={obs_ratio:.3f}"
            )

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"x_da_kdenudge_1d_seed{args.seed}.npz")

    np.savez(
        save_path,
        times=times,
        x_true=x_true_all[: n_steps + 1],
        x_obs=x_obs_all[: n_steps + 1],
        mask_missing=mask_all[: n_steps + 1],
        x_est=x_est_all,
        mesh_err=mesh_err,
        rmse=rmse,
        V=0.5 * rmse * rmse,
        mesh=z,
        config={
            **vars(args),
            "lambda_nudge": lambda_nudge,
        },
    )

    print(f"Saved DA result to: {save_path}")
    print(f"Final RMSE: {rmse[-1]:.6f}")
    print(f"Mean RMSE : {np.mean(rmse):.6f}")


def parse_args():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_path = os.path.normpath(
        os.path.join(this_dir, "../simulation/data/mv_sim_seed0.npz")
    )
    default_outdir = os.path.join(this_dir, "data")

    parser = argparse.ArgumentParser(
        description="1D Data Assimilation for McKean-Vlasov dynamics using mesh KDE nudging."
    )

    parser.add_argument("--data_path", type=str, default=default_data_path)
    parser.add_argument("--outdir", type=str, default=default_outdir)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.8)
    parser.add_argument("--sigma", type=float, default=1.0)

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
        help="If true, initialize the DA trajectory from x_true(0) instead of noisy observations.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    run_data_assimilation(args)


if __name__ == "__main__":
    main()
