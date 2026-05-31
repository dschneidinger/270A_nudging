#!/usr/bin/env python3
"""1D-1V electrostatic PIC for two-stream instability with DA-ready outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np


@dataclass
class SimulationConfig:
    num_particles: int = 1000
    num_grid: int = 128
    num_v_grid: int = 64
    length: float = 4.0 * np.pi
    t_end: float = 50.0
    dt: float = 0.05
    epsilon: float = 0.01
    mode: int = 1
    vmax: float = 6.0
    u0: float = 1.0
    sigma: float = 0.2
    seed: int = 0
    output_dir: str = "data"
    random_load: bool = False


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-particles", type=int, default=1000)
    parser.add_argument("--num-grid", type=int, default=128)
    parser.add_argument("--num-v-grid", type=int, default=64)
    parser.add_argument("--length", type=float, default=4.0 * np.pi)
    parser.add_argument("--t-end", type=float, default=50.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--mode", type=int, default=1)
    parser.add_argument("--vmax", type=float, default=6.0)
    parser.add_argument("--u0", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument(
        "--random-load",
        action="store_true",
        help="Use Monte Carlo loading instead of the quieter deterministic loader.",
    )
    args = parser.parse_args()
    return SimulationConfig(
        num_particles=args.num_particles,
        num_grid=args.num_grid,
        num_v_grid=args.num_v_grid,
        length=args.length,
        t_end=args.t_end,
        dt=args.dt,
        epsilon=args.epsilon,
        mode=args.mode,
        vmax=args.vmax,
        u0=args.u0,
        sigma=args.sigma,
        seed=args.seed,
        output_dir=args.output_dir,
        random_load=args.random_load,
    )


def physical_wavenumber(length: float, mode: int) -> float:
    return 2.0 * np.pi * mode / length


def quiet_start_positions(
    num_particles: int, length: float, epsilon: float, mode: int
) -> np.ndarray:
    """Deterministic loading of x using the exact cumulative density."""
    if abs(epsilon) >= 1.0:
        raise ValueError("quiet-start loading requires |epsilon| < 1.")

    k = physical_wavenumber(length, mode)
    u = (np.arange(num_particles, dtype=np.float64) + 0.5) / num_particles
    x = length * u

    if epsilon != 0.0:
        for _ in range(8):
            residual = x / length + epsilon * np.sin(k * x) / (length * k) - u
            derivative = (1.0 + epsilon * np.cos(k * x)) / length
            x -= residual / derivative

    return np.mod(x, length)


def mixture_branch_counts(num_particles: int) -> tuple[int, int]:
    n_minus = num_particles // 2
    n_plus = num_particles - n_minus
    return n_minus, n_plus


def quiet_start_truncated_normal(
    num_particles: int, mu: float, sigma: float, vmax: float
) -> np.ndarray:
    normal = NormalDist(mu=mu, sigma=sigma)
    cdf_low = normal.cdf(-vmax)
    cdf_high = normal.cdf(vmax)
    u = (np.arange(num_particles, dtype=np.float64) + 0.5) / num_particles
    q = cdf_low + (cdf_high - cdf_low) * u
    return np.array([normal.inv_cdf(value) for value in q], dtype=np.float64)


def quiet_start_velocities(
    num_particles: int, u0: float, sigma: float, vmax: float, rng: np.random.Generator
) -> np.ndarray:
    """Deterministic truncated bi-Maxwellian loading via inverse CDF."""
    n_minus, n_plus = mixture_branch_counts(num_particles)
    v_minus = quiet_start_truncated_normal(n_minus, -u0, sigma, vmax)
    v_plus = quiet_start_truncated_normal(n_plus, u0, sigma, vmax)
    velocities = np.concatenate((v_minus, v_plus))
    rng.shuffle(velocities)
    return velocities


def random_positions(
    num_particles: int,
    length: float,
    epsilon: float,
    mode: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample x from p(x) proportional to 1 + epsilon cos(kx) on [0, length)."""
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
        keep = y_trial < (1.0 + epsilon * np.cos(k * x_trial))
        accepted_batch = x_trial[keep]
        if accepted_batch.size:
            take = min(remaining, accepted_batch.size)
            accepted.append(accepted_batch[:take])
            remaining -= take

    return np.concatenate(accepted)


def random_truncated_normal(
    num_particles: int, mu: float, sigma: float, vmax: float, rng: np.random.Generator
) -> np.ndarray:
    accepted = []
    remaining = num_particles

    while remaining > 0:
        batch = max(2 * remaining, 1024)
        v_trial = rng.normal(loc=mu, scale=sigma, size=batch)
        keep = np.abs(v_trial) <= vmax
        accepted_batch = v_trial[keep]
        if accepted_batch.size:
            take = min(remaining, accepted_batch.size)
            accepted.append(accepted_batch[:take])
            remaining -= take

    return np.concatenate(accepted)


def random_velocities(
    num_particles: int, u0: float, sigma: float, vmax: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a truncated bi-Maxwellian with means +/-u0 and std sigma."""
    n_minus, n_plus = mixture_branch_counts(num_particles)
    v_minus = random_truncated_normal(n_minus, -u0, sigma, vmax, rng)
    v_plus = random_truncated_normal(n_plus, u0, sigma, vmax, rng)
    velocities = np.concatenate((v_minus, v_plus))
    rng.shuffle(velocities)
    return velocities


def load_particles(
    cfg: SimulationConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if cfg.random_load:
        xp = random_positions(
            cfg.num_particles, cfg.length, cfg.epsilon, cfg.mode, rng
        )
        vp = random_velocities(cfg.num_particles, cfg.u0, cfg.sigma, cfg.vmax, rng)
    else:
        xp = quiet_start_positions(
            cfg.num_particles, cfg.length, cfg.epsilon, cfg.mode
        )
        vp = quiet_start_velocities(
            cfg.num_particles, cfg.u0, cfg.sigma, cfg.vmax, rng
        )
    return xp, vp


def cic_deposit_density(
    xp: np.ndarray, particle_weight: float, num_grid: int, length: float
) -> np.ndarray:
    dx = length / num_grid
    grid_coord = xp / dx
    left = np.floor(grid_coord).astype(np.int64) % num_grid
    frac = grid_coord - np.floor(grid_coord)
    right = (left + 1) % num_grid

    rho = np.zeros(num_grid, dtype=np.float64)
    np.add.at(rho, left, particle_weight * (1.0 - frac) / dx)
    np.add.at(rho, right, particle_weight * frac / dx)
    return rho


def cic_gather_field(xp: np.ndarray, field_grid: np.ndarray, length: float) -> np.ndarray:
    num_grid = field_grid.size
    dx = length / num_grid
    grid_coord = xp / dx
    left = np.floor(grid_coord).astype(np.int64) % num_grid
    frac = grid_coord - np.floor(grid_coord)
    right = (left + 1) % num_grid
    return (1.0 - frac) * field_grid[left] + frac * field_grid[right]


def solve_field_periodic(rho: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Solve periodic Poisson with zero-mean potential."""
    num_grid = rho.size
    dx = length / num_grid
    source = rho - 1.0

    k = 2.0 * np.pi * np.fft.fftfreq(num_grid, d=dx)
    source_hat = np.fft.fft(source)

    phi_hat = np.zeros_like(source_hat, dtype=np.complex128)
    electric_hat = np.zeros_like(source_hat, dtype=np.complex128)

    nonzero = k != 0.0
    phi_hat[nonzero] = source_hat[nonzero] / (k[nonzero] ** 2)
    electric_hat[nonzero] = -1j * source_hat[nonzero] / k[nonzero]

    phi = np.fft.ifft(phi_hat).real
    electric = np.fft.ifft(electric_hat).real
    return phi, electric


def phase_space_grids(
    num_grid_x: int, num_grid_v: int, length: float, vmax: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    dx = length / num_grid_x
    dv = 2.0 * vmax / num_grid_v
    x_grid = np.arange(num_grid_x, dtype=np.float64) * dx
    v_grid = -vmax + np.arange(num_grid_v, dtype=np.float64) * dv
    return x_grid, v_grid, dx, dv


def deposit_phase_density(
    xp: np.ndarray,
    vp: np.ndarray,
    particle_weight: float,
    num_grid_x: int,
    num_grid_v: int,
    length: float,
    vmax: float,
) -> np.ndarray:
    """Deposit rho(t, x, v) on a tensor-product (x, v) mesh."""
    _, _, dx, dv = phase_space_grids(num_grid_x, num_grid_v, length, vmax)

    x_coord = xp / dx
    x_left = np.floor(x_coord).astype(np.int64) % num_grid_x
    x_frac = x_coord - np.floor(x_coord)
    x_right = (x_left + 1) % num_grid_x

    v_hi = np.nextafter(vmax, -np.inf)
    v_clipped = np.clip(vp, -vmax, v_hi)
    v_coord = (v_clipped + vmax) / dv
    v_left = np.floor(v_coord).astype(np.int64)
    v_frac = v_coord - np.floor(v_coord)
    v_right = np.minimum(v_left + 1, num_grid_v - 1)
    v_frac = np.where(v_right == v_left, 0.0, v_frac)

    weight = particle_weight / (dx * dv)
    rho_xv = np.zeros((num_grid_x, num_grid_v), dtype=np.float64)
    np.add.at(rho_xv, (x_left, v_left), weight * (1.0 - x_frac) * (1.0 - v_frac))
    np.add.at(rho_xv, (x_left, v_right), weight * (1.0 - x_frac) * v_frac)
    np.add.at(rho_xv, (x_right, v_left), weight * x_frac * (1.0 - v_frac))
    np.add.at(rho_xv, (x_right, v_right), weight * x_frac * v_frac)
    return rho_xv


def mode_amplitude(field_grid: np.ndarray, mode: int) -> float:
    coeffs = np.fft.fft(field_grid) / field_grid.size
    return 2.0 * np.abs(coeffs[mode])


def total_energies(vp: np.ndarray, electric: np.ndarray, length: float) -> tuple[float, float]:
    kinetic = 0.5 * np.mean(vp**2)
    dx = length / electric.size
    field = 0.5 * np.sum(electric**2) * dx / length
    return kinetic, field


def write_summary(
    output_dir: Path, cfg: SimulationConfig, e_mode: np.ndarray, save_name: str
) -> None:
    k = physical_wavenumber(cfg.length, cfg.mode)
    lines = [
        "1D-1V electrostatic PIC two-stream instability",
        f"saved_file = {save_name}",
        f"num_particles = {cfg.num_particles}",
        f"num_grid = {cfg.num_grid}",
        f"num_v_grid = {cfg.num_v_grid}",
        f"length = {cfg.length}",
        f"mode = {cfg.mode}",
        f"physical_k = {k}",
        f"epsilon = {cfg.epsilon}",
        f"u0 = {cfg.u0}",
        f"sigma = {cfg.sigma}",
        f"dt = {cfg.dt}",
        f"t_end = {cfg.t_end}",
        f"vmax = {cfg.vmax}",
        f"seed = {cfg.seed}",
        f"random_load = {cfg.random_load}",
        f"initial_E_mode = {e_mode[0]:.8e}",
        f"final_E_mode = {e_mode[-1]:.8e}",
        "",
        "Notes:",
        "- The output npz contains full particle trajectories states[t, n, :] = (x, v).",
        "- phase_density stores rho(t, x, v) on a tensor-product mesh.",
        "- The initial density is 0.5 f_M,sigma(v-u0) + 0.5 f_M,sigma(v+u0), multiplied by (1 + epsilon cos(kx)).",
        "- Poisson is solved for rho_x - 1 with a zero-mean potential.",
        "- To realize physical k = 0.5, use length = 4*pi and mode = 1.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_make_plot(output_dir: Path, times: np.ndarray, e_mode: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.semilogy(times, np.maximum(e_mode, 1.0e-16), linewidth=2.0)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|E_k(t)|$")
    ax.set_title("PIC Two-Stream Instability")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "E_mode.png", dpi=160)
    plt.close(fig)


def run_simulation(cfg: SimulationConfig) -> Path:
    if cfg.num_particles <= 0:
        raise ValueError("num_particles must be positive.")
    if cfg.num_grid < 4:
        raise ValueError("num_grid must be at least 4.")
    if cfg.num_v_grid < 4:
        raise ValueError("num_v_grid must be at least 4.")
    if cfg.mode <= 0:
        raise ValueError("mode must be a positive integer.")
    if cfg.mode >= cfg.num_grid // 2:
        raise ValueError("mode must be smaller than num_grid/2 for diagnostics.")
    if cfg.length <= 0.0:
        raise ValueError("length must be positive.")
    if cfg.dt <= 0.0:
        raise ValueError("dt must be positive.")
    if cfg.t_end <= 0.0:
        raise ValueError("t_end must be positive.")
    if cfg.vmax <= 0.0:
        raise ValueError("vmax must be positive.")
    if cfg.sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if cfg.u0 < 0.0:
        raise ValueError("u0 must be nonnegative.")
    if abs(cfg.epsilon) >= 1.0:
        raise ValueError("epsilon must satisfy |epsilon| < 1.")

    rng = np.random.default_rng(cfg.seed)
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(cfg.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    num_steps = int(np.round(cfg.t_end / cfg.dt))
    if not np.isclose(num_steps * cfg.dt, cfg.t_end):
        raise ValueError("t_end must be an integer multiple of dt.")

    x_grid, v_grid, _, _ = phase_space_grids(
        cfg.num_grid, cfg.num_v_grid, cfg.length, cfg.vmax
    )
    particle_weight = cfg.length / cfg.num_particles

    times = np.linspace(0.0, cfg.t_end, num_steps + 1, dtype=np.float64)
    states = np.empty((num_steps + 1, cfg.num_particles, 2), dtype=np.float64)
    rho_x_paths = np.empty((num_steps + 1, cfg.num_grid), dtype=np.float64)
    electric_paths = np.empty((num_steps + 1, cfg.num_grid), dtype=np.float64)
    phase_density = np.empty(
        (num_steps + 1, cfg.num_grid, cfg.num_v_grid), dtype=np.float64
    )
    e_mode = np.empty(num_steps + 1, dtype=np.float64)
    kinetic_energy = np.empty(num_steps + 1, dtype=np.float64)
    field_energy = np.empty(num_steps + 1, dtype=np.float64)

    xp, vp = load_particles(cfg, rng)
    rho = cic_deposit_density(xp, particle_weight, cfg.num_grid, cfg.length)
    _, electric = solve_field_periodic(rho, cfg.length)

    def record(step: int, xp_now: np.ndarray, vp_now: np.ndarray) -> None:
        states[step, :, 0] = xp_now
        states[step, :, 1] = vp_now
        rho_x_paths[step] = rho
        electric_paths[step] = electric
        phase_density[step] = deposit_phase_density(
            xp_now,
            vp_now,
            particle_weight,
            cfg.num_grid,
            cfg.num_v_grid,
            cfg.length,
            cfg.vmax,
        )
        e_mode[step] = mode_amplitude(electric, cfg.mode)
        kinetic_energy[step], field_energy[step] = total_energies(
            vp_now, electric, cfg.length
        )

    record(0, xp, vp)

    for step in range(1, num_steps + 1):
        e_particle = cic_gather_field(xp, electric, cfg.length)
        v_half = vp + 0.5 * cfg.dt * e_particle
        xp = (xp + cfg.dt * v_half) % cfg.length

        rho = cic_deposit_density(xp, particle_weight, cfg.num_grid, cfg.length)
        _, electric = solve_field_periodic(rho, cfg.length)

        e_particle = cic_gather_field(xp, electric, cfg.length)
        vp = v_half + 0.5 * cfg.dt * e_particle

        record(step, xp, vp)

    save_path = output_dir / f"mv_sim_seed{cfg.seed}.npz"
    np.savez(
        save_path,
        times=times,
        states=states,
        x_paths=states[..., 0],
        v_paths=states[..., 1],
        rho_x=rho_x_paths,
        electric=electric_paths,
        phase_density=phase_density,
        x_grid=x_grid,
        v_grid=v_grid,
        E_mode=e_mode,
        kinetic_energy=kinetic_energy,
        field_energy=field_energy,
        config={
            "num_particles": cfg.num_particles,
            "num_grid": cfg.num_grid,
            "num_v_grid": cfg.num_v_grid,
            "length": cfg.length,
            "t_end": cfg.t_end,
            "dt": cfg.dt,
            "epsilon": cfg.epsilon,
            "mode": cfg.mode,
            "physical_k": physical_wavenumber(cfg.length, cfg.mode),
            "vmax": cfg.vmax,
            "u0": cfg.u0,
            "sigma": cfg.sigma,
            "seed": cfg.seed,
            "random_load": cfg.random_load,
            "distribution": "two_stream",
        },
    )

    write_summary(output_dir, cfg, e_mode, save_path.name)
    maybe_make_plot(output_dir, times, e_mode)
    return save_path


def main() -> None:
    cfg = parse_args()
    save_path = run_simulation(cfg)
    print(f"Saved simulation to: {save_path}")


if __name__ == "__main__":
    main()
