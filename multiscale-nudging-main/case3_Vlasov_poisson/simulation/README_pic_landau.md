# 1D-1V PIC Landau Damping

Main script: `pic_landau_damping.py`

This example implements a periodic 1D-1V electrostatic PIC solver for
Landau damping:

$$
\partial_t f + v \partial_x f + E \partial_v f = 0, \\
-\partial_{xx} \phi = \rho_x - 1, \\
E = -\partial_x \phi.
$$

The default setup is chosen to be directly usable by the data assimilation
pipeline in `../assimilation`:

- spatial domain `x in [0, 4*pi)` so that the physical perturbation wavenumber
  can be `k = 0.5` with `mode = 1`
- initial condition `f(x,v,0) = f_M(v) * (1 + epsilon cos(kx))`
- default `num_particles = 1000`
- default velocity samples follow a truncated Maxwellian on `[-vmax, vmax]`

## Run

```bash
python pic_landau_damping.py
```

The provided `run_me` now uses a larger truth system:

```bash
python pic_landau_damping.py \
  --seed 0 \
  --num-particles 10000 \
  --num-grid 128 \
  --num-v-grid 64 \
  --t-end 50 \
  --dt 0.05 \
  --length 12.566370614359172 \
  --mode 1 \
  --epsilon 0.5 \
  --vmax 6.0 \
  --output-dir data
```

## Outputs

The script writes into `output_dir`:

- `mv_sim_seed<seed>.npz`
- `summary.txt`
- `E_mode.png` if `matplotlib` is available

`mv_sim_seed<seed>.npz` contains:

- `times`
- `states` with shape `(T, N, 2)` and `states[..., 0] = x`, `states[..., 1] = v`
- `x_paths`
- `v_paths`
- `rho_x`
- `electric`
- `phase_density`, which is the saved `rho(t, x, v)` array
- `x_grid`
- `v_grid`
- `E_mode`
- `kinetic_energy`
- `field_energy`
- `config`

## Numerical method

- particle loading: deterministic quiet start by default
- optional random loading: `--random-load`
- charge deposition in `x`: CIC
- phase-space density output `rho(t, x, v)`: tensor-product linear deposition
- field interpolation: CIC
- Poisson solve: periodic FFT with zero-mean potential
- time stepping: kick-drift-kick leapfrog

The assimilation script in `../assimilation` can now use a different particle
count from the truth simulation by setting `--num_particles` and optionally
`--obs_num_particles`.
