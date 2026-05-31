import argparse
import os
import numpy as np


def simulate_mckean_vlasov(
    n_particles: int,
    t_final: float,
    dt: float,
    sigma: float,
    rho: float,
    beta: float,
    seed: int,
    x0_mean: float,
    x0_std: float,
    y0_mean: float,
    y0_std: float,
    z0_mean: float,
    z0_std: float,
):
    """
    Simulate the 3D Lorenz-type McKean-Vlasov particle system
        dX_t = sigma (m_{t,y} - X_t) dt + sigma dW_t^x,
        dY_t = [m_{t,x} (rho - m_{t,z}) - Y_t] dt + sigma dW_t^y,
        dZ_t = [m_{t,x} m_{t,y} - beta Z_t] dt + sigma dW_t^z,
    where m_{t,x}, m_{t,y}, m_{t,z} are empirical particle means.

    The Brownian motion is interpreted componentwise, with independent
    Gaussian increments for X, Y, Z at each particle.
    """
    if dt <= 0:
        raise ValueError('dt must be positive')
    if t_final <= 0:
        raise ValueError('t_final must be positive')
    if n_particles <= 0:
        raise ValueError('n_particles must be positive')
    if x0_std < 0 or y0_std < 0 or z0_std < 0:
        raise ValueError('initial standard deviations must be nonnegative')

    n_steps = int(np.round(t_final / dt))
    if n_steps < 1:
        raise ValueError('t_final/dt must be at least 1 step')
    if not np.isclose(n_steps * dt, t_final):
        raise ValueError('t_final must be an integer multiple of dt')

    rng = np.random.default_rng(seed)
    x = rng.normal(loc=x0_mean, scale=x0_std, size=n_particles)
    y = rng.normal(loc=y0_mean, scale=y0_std, size=n_particles)
    z = rng.normal(loc=z0_mean, scale=z0_std, size=n_particles)

    times = dt * np.arange(n_steps + 1, dtype=float)
    x_paths = np.empty((n_steps + 1, n_particles), dtype=float)
    y_paths = np.empty((n_steps + 1, n_particles), dtype=float)
    z_paths = np.empty((n_steps + 1, n_particles), dtype=float)
    mx_paths = np.empty(n_steps + 1, dtype=float)
    my_paths = np.empty(n_steps + 1, dtype=float)
    mz_paths = np.empty(n_steps + 1, dtype=float)

    x_paths[0] = x
    y_paths[0] = y
    z_paths[0] = z
    mx_paths[0] = np.mean(x)
    my_paths[0] = np.mean(y)
    mz_paths[0] = np.mean(z)

    sqrt_dt = np.sqrt(dt)
    for k in range(n_steps):
        m_x = np.mean(x)
        m_y = np.mean(y)
        m_z = np.mean(z)
        noise =  sqrt_dt * rng.standard_normal((n_particles, 3))

        x_new = x + dt * sigma * (m_y - x) + noise[:, 0]
        y_new = y + dt * (m_x * (rho - m_z) - y) + noise[:, 1]
        z_new = z + dt * (m_x * m_y - beta * z) + noise[:, 2]

        x, y, z = x_new, y_new, z_new
        x_paths[k + 1] = x
        y_paths[k + 1] = y
        z_paths[k + 1] = z
        mx_paths[k + 1] = np.mean(x)
        my_paths[k + 1] = np.mean(y)
        mz_paths[k + 1] = np.mean(z)

    return times, x_paths, y_paths, z_paths, mx_paths, my_paths, mz_paths


def main():
    parser = argparse.ArgumentParser(
        description='Simulation for the 3D Lorenz-type McKean-Vlasov particle system.'
    )
    parser.add_argument('--n_particles', type=int, default=2000)
    parser.add_argument('--t_final', type=float, default=10.0)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--rho', type=float, default=28.0)
    parser.add_argument('--beta', type=float, default=8.0 / 3.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--x0_mean', type=float, default=0.0)
    parser.add_argument('--x0_std', type=float, default=1.0)
    parser.add_argument('--y0_mean', type=float, default=0.0)
    parser.add_argument('--y0_std', type=float, default=1.0)
    parser.add_argument('--z0_mean', type=float, default=0.0)
    parser.add_argument('--z0_std', type=float, default=1.0)
    parser.add_argument('--outdir', type=str, default='data')
    args = parser.parse_args()

    times, x_paths, y_paths, z_paths, mx_paths, my_paths, mz_paths = simulate_mckean_vlasov(
        n_particles=args.n_particles,
        t_final=args.t_final,
        dt=args.dt,
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
        seed=args.seed,
        x0_mean=args.x0_mean,
        x0_std=args.x0_std,
        y0_mean=args.y0_mean,
        y0_std=args.y0_std,
        z0_mean=args.z0_mean,
        z0_std=args.z0_std,
    )

    means = np.column_stack((mx_paths, my_paths, mz_paths))
    states = np.stack((x_paths, y_paths, z_paths), axis=-1)

    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f'mv_sim_seed{args.seed}.npz')

    np.savez(
        save_path,
        times=times,
        states=states,
        means=means,
        x_paths=x_paths,
        y_paths=y_paths,
        z_paths=z_paths,
        mx_paths=mx_paths,
        my_paths=my_paths,
        mz_paths=mz_paths,
        config={
            'n_particles': args.n_particles,
            't_final': args.t_final,
            'dt': args.dt,
            'sigma': args.sigma,
            'rho': args.rho,
            'beta': args.beta,
            'seed': args.seed,
            'x0_mean': args.x0_mean,
            'x0_std': args.x0_std,
            'y0_mean': args.y0_mean,
            'y0_std': args.y0_std,
            'z0_mean': args.z0_mean,
            'z0_std': args.z0_std,
        },
    )

    print(f'Saved simulation to: {save_path}')
    print(
        'Final empirical means: '
        f'mx(T)={mx_paths[-1]:.6f}, '
        f'my(T)={my_paths[-1]:.6f}, '
        f'mz(T)={mz_paths[-1]:.6f}'
    )


if __name__ == '__main__':
    main()
