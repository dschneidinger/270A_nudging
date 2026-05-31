import argparse
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np


EPS = 1e-14


def parse_float_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return np.asarray(values, dtype=float)


def parse_int_list(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return np.asarray(values, dtype=int)


def safe_window_size(length, fraction, minimum):
    return min(length, max(minimum, int(np.ceil(fraction * length))))


def estimate_long_time_plateau(V, tail_fraction):
    tail_n = safe_window_size(len(V), tail_fraction, minimum=5)
    return float(np.mean(V[-tail_n:])), tail_n


def estimate_head_mean(V, head_fraction, fit_start_fraction, tail_n):
    n = len(V)
    head_n = safe_window_size(n, head_fraction, minimum=5)
    start = min(n - 1, max(0, int(np.floor(fit_start_fraction * (n - 1)))))
    stop = min(n - tail_n, start + head_n)
    if stop <= start:
        start = 0
        stop = min(n, head_n)
    return float(np.mean(V[start:stop]))


def estimate_convergence_rate(times, V, V_inf, fit_start_fraction, fit_stop_fraction):
    n = len(V)
    start = min(n - 2, max(1, int(np.floor(fit_start_fraction * (n - 1)))))
    stop = min(n, max(start + 4, int(np.ceil(fit_stop_fraction * n))))
    fit_idx = np.arange(start, stop)

    amplitude = max(float(np.max(V[fit_idx]) - V_inf), 0.0)
    margin = max(EPS, 0.05 * amplitude)
    fit_idx = fit_idx[V[fit_idx] > V_inf + margin]
    if fit_idx.size < 4:
        return np.nan, np.nan, 0

    log_energy = np.log(np.maximum(V[fit_idx] - V_inf, EPS))
    slope, _ = np.polyfit(times[fit_idx], log_energy, 1)
    alpha_hat = float(max(0.0, -slope))
    return alpha_hat, float(slope), int(fit_idx.size)


def summarize_result(result_path, args):
    data = np.load(result_path, allow_pickle=True)
    times = np.asarray(data["times"], dtype=float)
    if "V" in data.files:
        V = np.asarray(data["V"], dtype=float)
    else:
        rmse = np.asarray(data["rmse"], dtype=float)
        V = 0.5 * rmse * rmse

    plateau, tail_n = estimate_long_time_plateau(V, args.tail_fraction)
    head_mean = estimate_head_mean(V, args.head_fraction, args.fit_start_fraction, tail_n)
    alpha_hat, slope, fit_points = estimate_convergence_rate(
        times,
        V,
        V_inf=plateau,
        fit_start_fraction=args.fit_start_fraction,
        fit_stop_fraction=args.fit_stop_fraction,
    )
    plateau_ratio = plateau / max(head_mean, EPS)
    converged = bool(
        np.isfinite(alpha_hat)
        and alpha_hat > args.rate_tol
        and plateau_ratio < args.convergence_ratio
    )

    return {
        "plateau_V": plateau,
        "head_V": head_mean,
        "plateau_ratio": plateau_ratio,
        "alpha_hat": alpha_hat,
        "log_slope": slope,
        "fit_points": fit_points,
        "converged": converged,
        "final_V": float(V[-1]),
    }


def format_value_token(value):
    return f"{value:.3e}".replace("+", "").replace(".", "p")


def fit_affine_threshold_form(lambda_values, alpha_values):
    mask = np.isfinite(alpha_values) & (alpha_values > 0.0)
    if np.count_nonzero(mask) < 2:
        return None
    x = lambda_values[mask]
    y = alpha_values[mask]
    if np.ptp(x) < EPS:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    lambda_star = np.nan
    if slope > EPS:
        lambda_star = float(-intercept / slope)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "lambda_star": lambda_star,
    }


def fit_delta_floor(delta_values, plateau_values):
    x = delta_values * delta_values
    y = plateau_values
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return None
    x = x[mask]
    y = y[mask]
    if np.ptp(x) < EPS:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot < EPS else 1.0 - ss_res / ss_tot
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
    }


def maybe_get_theory_prediction(lambda_values, args):
    required = [args.theory_c_adv, args.theory_kappa, args.theory_nu, args.theory_c_p]
    if any(value is None for value in required):
        return None

    c_adv = float(args.theory_c_adv)
    kappa = float(args.theory_kappa)
    nu = float(args.theory_nu)
    c_p = float(args.theory_c_p)
    raw_alpha = (kappa + nu * lambda_values) / c_p - (2.0 * c_adv * c_adv) / kappa
    theory = {
        "alpha_raw": raw_alpha,
        "alpha_clipped": np.maximum(raw_alpha, 0.0),
        "lambda_star": (2.0 * c_adv * c_adv * c_p / kappa - kappa) / nu,
    }
    if args.theory_delta_h is not None and args.theory_rho_upper is not None:
        theory["h_condition_ok"] = bool(
            args.theory_delta_h <= nu / (2.0 * float(args.theory_rho_upper))
        )
    return theory


def get_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def configure_lambda_axis(ax, lambda_values):
    positive = lambda_values[lambda_values > 0.0]
    if np.any(lambda_values <= 0.0) and positive.size:
        ax.set_xscale("symlog", linthresh=float(np.min(positive)))
    elif positive.size >= 2 and (np.max(positive) / np.min(positive) > 50.0):
        ax.set_xscale("log")


def save_csv(path, header, array):
    np.savetxt(path, array, delimiter=",", header=header, comments="")


def save_parameter_grid_csv(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("a,nudge_iters,outdir,summary_path\n")
        for record in records:
            handle.write(
                f"{record['a']:.16g},{record['nudge_iters']},{record['outdir']},{record['summary_path']}\n"
            )


def clone_args(args, **updates):
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def parameter_case_label(a_value, nudge_iters):
    return f"a_{format_value_token(a_value)}__nudge_iters_{int(nudge_iters):03d}"


def iter_parameter_cases(args):
    for a_value, nudge_iters in product(args.a_values, args.nudge_iter_values):
        yield float(a_value), int(nudge_iters)


def run_experiment(args):
    if args.experiment == "lambda_scan":
        return run_lambda_scan(args)
    if args.experiment == "delta_scan":
        return run_delta_scan(args)
    raise ValueError(f"Unsupported experiment type: {args.experiment}")


def plot_lambda_scan(lambda_values, summary, fit_info, theory_info, outdir, args):
    plt = get_matplotlib()
    order = np.argsort(lambda_values)
    x = lambda_values[order]
    converged = summary["converged"][order].astype(float)
    alpha_hat = summary["alpha_hat"][order]

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.0), sharex=True)

    axes[0].plot(x, converged, "o-", color="tab:blue", lw=1.5, ms=5)
    axes[0].set_ylabel("Converged")
    axes[0].set_yticks([0.0, 1.0])
    axes[0].set_yticklabels(["No", "Yes"])
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(f"Fixed h={args.h_bw:g}: convergence diagnostics vs lambda")

    axes[1].plot(x, alpha_hat, "o", color="tab:orange", ms=6, label="estimated rate")
    if fit_info is not None:
        x_plot = np.linspace(np.min(x), np.max(x), 300)
        y_fit = fit_info["slope"] * x_plot + fit_info["intercept"]
        axes[1].plot(x_plot, np.maximum(y_fit, 0.0), "-", color="black", lw=1.4, label="affine threshold-form fit")
        if np.isfinite(fit_info["lambda_star"]):
            axes[1].axvline(
                fit_info["lambda_star"],
                color="black",
                linestyle="--",
                lw=1.0,
                label=f"fit threshold={fit_info['lambda_star']:.3e}",
            )
    if theory_info is not None:
        x_plot = np.linspace(np.min(x), np.max(x), 300)
        alpha_theory = np.maximum(
            (args.theory_kappa + args.theory_nu * x_plot) / args.theory_c_p
            - 2.0 * args.theory_c_adv * args.theory_c_adv / args.theory_kappa,
            0.0,
        )
        axes[1].plot(x_plot, alpha_theory, "--", color="tab:green", lw=1.3, label="theory")
        axes[1].axvline(
            theory_info["lambda_star"],
            color="tab:green",
            linestyle=":",
            lw=1.1,
            label=f"theory threshold={theory_info['lambda_star']:.3e}",
        )

    axes[1].set_xlabel("lambda")
    axes[1].set_ylabel("Rate")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    configure_lambda_axis(axes[0], x)
    configure_lambda_axis(axes[1], x)

    fig.tight_layout()
    save_path = os.path.join(outdir, "lambda_scan_summary.png")
    fig.savefig(save_path, dpi=args.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved lambda scan plot to: {save_path}")


def plot_delta_scan(delta_values, plateau_values, fit_info, outdir, args):
    plt = get_matplotlib()
    x = delta_values * delta_values
    order = np.argsort(x)
    x = x[order]
    y = plateau_values[order]

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.0))
    ax.plot(x, y, "o", color="tab:red", ms=6, label="estimated V(infty)")
    if fit_info is not None:
        x_plot = np.linspace(np.min(x), np.max(x), 300)
        y_fit = fit_info["slope"] * x_plot + fit_info["intercept"]
        ax.plot(
            x_plot,
            y_fit,
            "-",
            color="black",
            lw=1.4,
            label=f"linear fit, R^2={fit_info['r2']:.3f}",
        )

    ax.set_xlabel("Delta^2")
    ax.set_ylabel("Estimated V(infty)")
    ax.set_title(
        f"Fixed lambda={resolve_lambda_for_scan(args):g}, h={args.h_bw:g}: floor vs Delta^2"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    fig.tight_layout()
    save_path = os.path.join(outdir, "delta_scan_summary.png")
    fig.savefig(save_path, dpi=args.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Delta scan plot to: {save_path}")


def resolve_lambda_for_scan(args):
    if args.lambda_nudge is None:
        return args.mu_nudge
    return args.lambda_nudge


def simulation_script_path():
    return Path(__file__).with_name("simulation.py")


def build_command(args, outdir, a_value, c_value, lambda_value):
    cmd = [
        sys.executable,
        str(simulation_script_path()),
        "--data_path",
        args.data_path,
        "--outdir",
        str(outdir),
        "--seed",
        str(args.seed),
        "--max_steps",
        str(args.max_steps),
        "--dt",
        str(args.dt),
        "--a",
        str(a_value),
        "--c",
        str(c_value),
        "--sigma",
        str(args.sigma),
        "--obs_keep_prob",
        str(args.obs_keep_prob),
        "--obs_noise_std",
        str(args.obs_noise_std),
        "--grid_n",
        str(args.grid_n),
        "--h_bw",
        str(args.h_bw),
        "--mu_nudge",
        str(args.mu_nudge),
        "--lambda_nudge",
        str(lambda_value),
        "--nudge_iters",
        str(args.nudge_iters),
        "--mesh_margin",
        str(args.mesh_margin),
        "--forecast_seed_offset",
        str(args.forecast_seed_offset),
    ]
    cmd.append("--init_from_truth" if args.init_from_truth else "--no-init_from_truth")
    return cmd


def run_single_case(args, run_dir, a_value, c_value, lambda_value):
    os.makedirs(run_dir, exist_ok=True)
    cmd = build_command(args, outdir=run_dir, a_value=a_value, c_value=c_value, lambda_value=lambda_value)
    subprocess.run(cmd, check=True)
    result_path = Path(run_dir) / f"x_da_kdenudge_1d_seed{args.seed}.npz"
    return summarize_result(result_path, args), result_path


def run_lambda_scan(args):
    lambda_values = parse_float_list(args.lambda_values)
    if lambda_values.size == 0:
        raise ValueError("lambda_scan requires at least one lambda value")

    scan_outdir = Path(args.outdir) / "lambda_scan"
    runs_outdir = scan_outdir / "runs"
    runs_outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for idx, lambda_value in enumerate(lambda_values, start=1):
        run_dir = runs_outdir / f"lambda_{idx:03d}_{format_value_token(lambda_value)}"
        print(f"[lambda_scan] {idx}/{len(lambda_values)} | lambda={lambda_value:.6g}")
        summary, _ = run_single_case(
            args,
            run_dir,
            a_value=args.a,
            c_value=args.c,
            lambda_value=lambda_value,
        )
        summaries.append(summary)

    summary_dict = {
        "converged": np.asarray([item["converged"] for item in summaries], dtype=bool),
        "alpha_hat": np.asarray([item["alpha_hat"] for item in summaries], dtype=float),
        "plateau_V": np.asarray([item["plateau_V"] for item in summaries], dtype=float),
        "head_V": np.asarray([item["head_V"] for item in summaries], dtype=float),
        "plateau_ratio": np.asarray([item["plateau_ratio"] for item in summaries], dtype=float),
        "final_V": np.asarray([item["final_V"] for item in summaries], dtype=float),
        "fit_points": np.asarray([item["fit_points"] for item in summaries], dtype=int),
    }

    fit_info = fit_affine_threshold_form(lambda_values, summary_dict["alpha_hat"])
    theory_info = maybe_get_theory_prediction(lambda_values, args)

    payload = {
        "lambda_values": lambda_values,
        "a_value": args.a,
        "c_value": args.c,
        **summary_dict,
        "config": np.array(vars(args), dtype=object),
    }
    if fit_info is not None:
        payload["fit_slope"] = fit_info["slope"]
        payload["fit_intercept"] = fit_info["intercept"]
        payload["fit_lambda_star"] = fit_info["lambda_star"]
    if theory_info is not None:
        payload["theory_alpha_raw"] = theory_info["alpha_raw"]
        payload["theory_alpha_clipped"] = theory_info["alpha_clipped"]
        payload["theory_lambda_star"] = theory_info["lambda_star"]
        if "h_condition_ok" in theory_info:
            payload["theory_h_condition_ok"] = theory_info["h_condition_ok"]

    summary_path = scan_outdir / "lambda_scan_summary.npz"
    np.savez(summary_path, **payload)
    print(f"Saved lambda scan summary to: {summary_path}")

    csv_array = np.column_stack(
        [
            lambda_values,
            summary_dict["converged"].astype(int),
            summary_dict["alpha_hat"],
            summary_dict["plateau_V"],
            summary_dict["head_V"],
            summary_dict["plateau_ratio"],
            summary_dict["final_V"],
            summary_dict["fit_points"],
        ]
    )
    save_csv(
        scan_outdir / "lambda_scan_summary.csv",
        "lambda,converged,alpha_hat,plateau_V,head_V,plateau_ratio,final_V,fit_points",
        csv_array,
    )
    plot_lambda_scan(lambda_values, summary_dict, fit_info, theory_info, scan_outdir, args)
    return scan_outdir / "lambda_scan_summary.npz"


def run_delta_scan(args):
    delta_values = parse_float_list(args.delta_values)
    if delta_values.size == 0:
        raise ValueError("delta_scan requires at least one Delta value")

    scan_outdir = Path(args.outdir) / "delta_scan"
    runs_outdir = scan_outdir / "runs"
    runs_outdir.mkdir(parents=True, exist_ok=True)

    summaries = []
    a_values = []
    c_values = []
    lambda_value = resolve_lambda_for_scan(args)

    for idx, delta_value in enumerate(delta_values, start=1):
        a_value = args.a + args.delta_a_scale * delta_value
        c_value = args.c + args.delta_c_scale * delta_value
        a_values.append(a_value)
        c_values.append(c_value)
        run_dir = runs_outdir / f"delta_{idx:03d}_{format_value_token(delta_value)}"
        print(
            f"[delta_scan] {idx}/{len(delta_values)} | Delta={delta_value:.6g} "
            f"-> a={a_value:.6g}, c={c_value:.6g}"
        )
        summary, _ = run_single_case(
            args,
            run_dir,
            a_value=a_value,
            c_value=c_value,
            lambda_value=lambda_value,
        )
        summaries.append(summary)

    a_values = np.asarray(a_values, dtype=float)
    c_values = np.asarray(c_values, dtype=float)
    plateau_values = np.asarray([item["plateau_V"] for item in summaries], dtype=float)
    alpha_values = np.asarray([item["alpha_hat"] for item in summaries], dtype=float)
    converged = np.asarray([item["converged"] for item in summaries], dtype=bool)
    final_V = np.asarray([item["final_V"] for item in summaries], dtype=float)

    fit_info = fit_delta_floor(delta_values, plateau_values)

    payload = {
        "delta_values": delta_values,
        "delta_sq": delta_values * delta_values,
        "a_values": a_values,
        "c_values": c_values,
        "plateau_V": plateau_values,
        "alpha_hat": alpha_values,
        "converged": converged,
        "final_V": final_V,
        "config": np.array(vars(args), dtype=object),
    }
    if fit_info is not None:
        payload["fit_slope"] = fit_info["slope"]
        payload["fit_intercept"] = fit_info["intercept"]
        payload["fit_r2"] = fit_info["r2"]

    summary_path = scan_outdir / "delta_scan_summary.npz"
    np.savez(summary_path, **payload)
    print(f"Saved Delta scan summary to: {summary_path}")

    csv_array = np.column_stack(
        [
            delta_values,
            delta_values * delta_values,
            a_values,
            c_values,
            plateau_values,
            final_V,
            alpha_values,
            converged.astype(int),
        ]
    )
    save_csv(
        scan_outdir / "delta_scan_summary.csv",
        "Delta,Delta_sq,a_value,c_value,plateau_V,final_V,alpha_hat,converged",
        csv_array,
    )
    plot_delta_scan(delta_values, plateau_values, fit_info, scan_outdir, args)
    return scan_outdir / "delta_scan_summary.npz"


def normalize_scan_args(args):
    args.a_values = parse_float_list(args.a)
    if args.a_values.size == 0:
        raise ValueError("--a must contain at least one value")

    args.nudge_iter_values = parse_int_list(args.nudge_iters)
    if args.nudge_iter_values.size == 0:
        raise ValueError("--nudge_iters must contain at least one value")

    args.a = float(args.a_values[0])
    args.nudge_iters = int(args.nudge_iter_values[0])
    return args


def validate_args(args):
    if args.h_bw <= 0.0:
        raise ValueError("h_bw must be positive")
    if np.any(args.nudge_iter_values < 1):
        raise ValueError("all nudge_iters values must be at least 1")
    if args.grid_n < 8:
        raise ValueError("grid_n must be at least 8")
    if not np.all(np.isfinite(args.a_values)):
        raise ValueError("all a values must be finite")
    for name in [
        "tail_fraction",
        "head_fraction",
        "fit_start_fraction",
        "fit_stop_fraction",
        "convergence_ratio",
    ]:
        value = getattr(args, name)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"{name} must be in (0, 1]")
    if args.fit_start_fraction >= args.fit_stop_fraction:
        raise ValueError("fit_start_fraction must be smaller than fit_stop_fraction")


def parse_args():
    this_dir = Path(__file__).resolve().parent
    default_data_path = this_dir.parent / "simulation" / "data" / "mv_sim_seed0.npz"
    default_outdir = this_dir / "data"

    parser = argparse.ArgumentParser(
        description="Sweep lambda or Delta by repeatedly calling simulation.py."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["lambda_scan", "delta_scan"],
    )
    parser.add_argument("--data_path", type=str, default=str(default_data_path))
    parser.add_argument("--outdir", type=str, default=str(default_outdir))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=-1)

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument(
        "--a",
        type=str,
        default="1.0",
        help="Single a value or comma-separated list of a values.",
    )
    parser.add_argument("--c", type=float, default=0.8)
    parser.add_argument("--sigma", type=float, default=1.0)

    parser.add_argument("--obs_keep_prob", type=float, default=0.3)
    parser.add_argument("--obs_noise_std", type=float, default=0.05)
    parser.add_argument("--grid_n", type=int, default=256)
    parser.add_argument("--h_bw", type=float, default=0.5)
    parser.add_argument("--mu_nudge", type=float, default=1e-4)
    parser.add_argument("--lambda_nudge", type=float, default=None)
    parser.add_argument(
        "--nudge_iters",
        type=str,
        default="20",
        help="Single value or comma-separated list of nudging iteration counts.",
    )
    parser.add_argument("--mesh_margin", type=float, default=2.0)
    parser.add_argument(
        "--enforce_obs",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--forecast_seed_offset",
        type=int,
        default=101,
    )
    parser.add_argument(
        "--init_from_truth",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument(
        "--lambda_values",
        type=str,
        default="0,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2",
    )
    parser.add_argument(
        "--delta_values",
        type=str,
        default="0,0.02,0.05,0.08,0.12,0.16,0.2",
    )
    parser.add_argument(
        "--delta_a_scale",
        type=float,
        default=1.0,
        help="delta_scan uses a_run = a + delta_a_scale * Delta.",
    )
    parser.add_argument(
        "--delta_c_scale",
        type=float,
        default=1.0,
        help="delta_scan uses c_run = c + delta_c_scale * Delta.",
    )

    parser.add_argument("--tail_fraction", type=float, default=0.25)
    parser.add_argument("--head_fraction", type=float, default=0.20)
    parser.add_argument("--fit_start_fraction", type=float, default=0.10)
    parser.add_argument("--fit_stop_fraction", type=float, default=0.75)
    parser.add_argument("--convergence_ratio", type=float, default=0.95)
    parser.add_argument("--rate_tol", type=float, default=1e-3)
    parser.add_argument("--plot_dpi", type=int, default=180)

    parser.add_argument("--theory_c_adv", type=float, default=None)
    parser.add_argument("--theory_kappa", type=float, default=None)
    parser.add_argument("--theory_nu", type=float, default=None)
    parser.add_argument("--theory_c_p", type=float, default=None)
    parser.add_argument("--theory_delta_h", type=float, default=None)
    parser.add_argument("--theory_rho_upper", type=float, default=None)

    args = normalize_scan_args(parser.parse_args())
    validate_args(args)
    return args


def main():
    args = parse_args()
    cases = list(iter_parameter_cases(args))
    if len(cases) == 1:
        run_experiment(args)
        return

    suite_outdir = Path(args.outdir) / f"{args.experiment}_parameter_grid"
    suite_outdir.mkdir(parents=True, exist_ok=True)

    records = []
    total_cases = len(cases)
    for idx, (a_value, nudge_iters) in enumerate(cases, start=1):
        case_outdir = suite_outdir / parameter_case_label(a_value, nudge_iters)
        print(
            f"[parameter_grid] {idx}/{total_cases} | "
            f"a={a_value:.6g}, nudge_iters={nudge_iters}"
        )
        run_args = clone_args(
            args,
            a=float(a_value),
            nudge_iters=int(nudge_iters),
            outdir=str(case_outdir),
        )
        summary_path = run_experiment(run_args)
        records.append(
            {
                "a": float(a_value),
                "nudge_iters": int(nudge_iters),
                "outdir": str(case_outdir),
                "summary_path": str(summary_path),
            }
        )

    grid_summary_path = suite_outdir / "parameter_grid_summary.csv"
    save_parameter_grid_csv(grid_summary_path, records)
    print(f"Saved parameter-grid summary to: {grid_summary_path}")


if __name__ == "__main__":
    main()
