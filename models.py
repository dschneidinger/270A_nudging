"""Model architectures for predicting the distribution function f from sparse phi."""

import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected MLP: flat [phi...] vector -> full f(x,v).

    Used by both the `mlp` (pair input) and `mlp_residual` experiments -- the
    difference between them is only how the dataset builds the input vector.
    """

    def __init__(
        self,
        n_sparse_measurements,
        n_x_grid,
        n_v_grid,
        hidden_layers=[256, 512, 1024, 512],
        dropout=0.1,
    ):
        super().__init__()

        # Input: sparse phi at t, sparse phi (or residual) at t-1, and dt = 2*n_sparse + 1
        input_size = 2 * n_sparse_measurements + 1
        output_size = n_x_grid * n_v_grid

        self.n_x_grid = n_x_grid
        self.n_v_grid = n_v_grid

        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)
        return out.view(-1, self.n_x_grid, self.n_v_grid)


class FCCNNDecoder(nn.Module):
    """FC encoder + CNN decoder for predicting a 2D spatial field from a small input vector.

    Why this architecture:
    - The input is a small 1d vector, so FC layers.
    - The output is a 2D spatial field (128x64) where neighboring pixels are
      correlated, so convolutional layers are the right tool to decode it.
    - No activation on the final layer since outputs are normalized regression targets.
    """

    def __init__(self, n_sparse_measurements, n_x_grid, n_v_grid, latent_dim=256):
        super().__init__()

        input_size = 2 * n_sparse_measurements + 1
        self.n_x_grid = n_x_grid
        self.n_v_grid = n_v_grid

        # FC encoder: sparse phi vector -> latent -> spatial seed (128, 4, 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 128 * 4 * 2),
            nn.SiLU(),
        )

        # CNN decoder: (128, 4, 2) -> (1, 128, 64) via 5 upsampling stages
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        latent = self.encoder(x)
        latent = latent.view(-1, 128, 4, 2)
        out = self.decoder(latent)
        return out.squeeze(1)  # (batch, n_x_grid, n_v_grid)


class Conv1D2DNet(nn.Module):
    """Conv1D encoder (temporal) -> FC bridge -> Conv2D decoder (spatial).

    Conv1D processes the sequence of sparse phi measurements across timesteps,
    learning temporal patterns (velocity, acceleration) directly.
    A small FC bridge reshapes the temporal features into a 2D spatial seed.
    Conv2D decoder upsamples the seed into the full (n_x, n_v) output grid.
    """

    def __init__(self, n_sparse_measurements, n_x_grid, n_v_grid, n_timesteps=5, latent_dim=256):
        super().__init__()

        self.n_x_grid = n_x_grid
        self.n_v_grid = n_v_grid

        # Input channels = n_sparse, sequence length = T+1 (T timesteps + dt row)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(n_sparse_measurements, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),  # pool across time -> (128, 1)
        )

        # FC bridge: temporal features -> spatial seed
        self.bridge = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, 128 * 4 * 2),
            nn.SiLU(),
        )

        # Conv2D decoder: (128, 4, 2) -> (1, 128, 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.SiLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # x: (batch, T+1, n_sparse) -> Conv1d wants (batch, channels, seq_len)
        x = x.transpose(1, 2)
        temporal = self.temporal_encoder(x)  # (batch, 128, 1)
        temporal = temporal.squeeze(2)       # (batch, 128)
        spatial_seed = self.bridge(temporal)
        spatial_seed = spatial_seed.view(-1, 128, 4, 2)
        out = self.decoder(spatial_seed)
        return out.squeeze(1)  # (batch, n_x_grid, n_v_grid)


# Model registry: maps a config `model_type` to its class + required input mode

MODEL_REGISTRY = {
    "mlp": {"class": MLP, "input_mode": "flat_pair"},
    "mlp_residual": {"class": MLP, "input_mode": "flat_residual"},
    "fc_cnn": {"class": FCCNNDecoder, "input_mode": "flat_pair"},
    "conv1d2d": {"class": Conv1D2DNet, "input_mode": "sequence"},
}


def get_input_mode(model_type):
    """Return the dataset input_mode required by a given model_type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose one of: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type]["input_mode"]


def build_model(model_type, n_sparse, n_x, n_v, config):
    """Instantiate the model selected by config['model_type'].

    Pulls only the config keys each architecture needs, so a single config dict
    can carry parameters for every model type.
    """
    cls = MODEL_REGISTRY[model_type]["class"]

    if model_type in ("mlp", "mlp_residual"):
        return cls(
            n_sparse_measurements=n_sparse,
            n_x_grid=n_x,
            n_v_grid=n_v,
            hidden_layers=config["hidden_layers"],
            dropout=config["dropout"],
        )
    elif model_type == "fc_cnn":
        return cls(
            n_sparse_measurements=n_sparse,
            n_x_grid=n_x,
            n_v_grid=n_v,
            latent_dim=config["latent_dim"],
        )
    elif model_type == "conv1d2d":
        return cls(
            n_sparse_measurements=n_sparse,
            n_x_grid=n_x,
            n_v_grid=n_v,
            n_timesteps=config["n_timesteps"],
            latent_dim=config["latent_dim"],
        )
    raise ValueError(f"Unknown model_type '{model_type}'")
