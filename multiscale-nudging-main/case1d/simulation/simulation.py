import argparse
import os
import numpy as np


def simulate_mckean_vlasov(
    n_particles: int,
    t_final: float,
    dt: float,
    a: float,
    c: float,
    sigma: float,
    seed: int,
    x0_mean: float,
    x0_std: float,
):
    """
    Simulate 1D McKean-Vlasov dynamics:
        dX_t = (-a X_t + c m_t) dt + sigma dW_t,
    with empirical mean m_t = (1/N) sum_i X_t^i.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    if t_final <= 0:
        raise ValueError("t_final must be positive")
    if n_particles <= 0:
        raise ValueError("n_particles must be positive")

    n_steps = int(np.round(t_final / dt))
    if n_steps < 1:
        raise ValueError("t_final/dt must be at least 1 step")

    rng = np.random.default_rng(seed)
    x = rng.normal(loc=x0_mean, scale=x0_std, size=n_particles)

    times = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x_paths = np.empty((n_steps + 1, n_particles), dtype=float)
    means = np.empty(n_steps + 1, dtype=float)

    x_paths[0] = x
    means[0] = np.mean(x)

    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        m = np.mean(x)
        drift = -a * (x - m)
        noise = sigma * sqrt_dt * rng.standard_normal(n_particles)
        x = x + dt * drift + noise

        x_paths[k + 1] = x
        means[k + 1] = np.mean(x)

    return times, x_paths, means


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Simulation for 1D McKean-Vlasov dynamics with linear mean-field coupling: "
            "b_true(x, mu) = -a x + c m(mu)."
        )
    )
    parser.add_argument("--n_particles", type=int, default=2000)
    parser.add_argument("--t_final", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.8)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x0_mean", type=float, default=0.0)
    parser.add_argument("--x0_std", type=float, default=1.0)
    parser.add_argument("--outdir", type=str, default="data")
    args = parser.parse_args()

    times, x_paths, means = simulate_mckean_vlasov(
        n_particles=args.n_particles,
        t_final=args.t_final,
        dt=args.dt,
        a=args.a,
        c=args.c,
        sigma=args.sigma,
        seed=args.seed,
        x0_mean=args.x0_mean,
        x0_std=args.x0_std,
    )

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"mv_sim_seed{args.seed}.npz")

    np.savez(
        save_path,
        times=times,
        x_paths=x_paths,
        means=means,
        config={
            "n_particles": args.n_particles,
            "t_final": args.t_final,
            "dt": args.dt,
            "a": args.a,
            "c": args.c,
            "sigma": args.sigma,
            "seed": args.seed,
            "x0_mean": args.x0_mean,
            "x0_std": args.x0_std,
        },
    )

    print(f"Saved simulation to: {save_path}")
    print(f"Final empirical mean m(T): {means[-1]:.6f}")


if __name__ == "__main__":
    main()
