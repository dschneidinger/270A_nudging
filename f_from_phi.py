# Train a Network to find a distribution function f, from a given potential phi, measured sparsely in space.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

r"""
Vlasov equation : \partial_t f + v \cdot \nabla_x f + F \cdot \nabla_v f = 0
For now we will just consider the electrostatic case, where F = q E = -\nabla_x * \phi * q #TODO check

Full distribution function f(x1,v1,t) is output from the numerical solver.
In the future this could be from osiris, but for now it will be from the numerical solver Hayden's postdoc wrote
"""

def phi_from_E(E, x_grid)-> np.ndarray:
    """Compute electric potential from electric field using integration."""
    phi = np.cumsum(-E * np.gradient(x_grid), axis=1)
    return phi


class VlasovDataset(Dataset):
    """Dataset for training MLP to predict f from sparse phi measurements."""
    def __init__(self, phi_sparse, f_full, time_diffs):
        """
        Args:
            #TODO, I think you need to include the positions of the sparse measurements
            phi_sparse: (N_samples, 2, N_sparse) - sparse phi at t and t-1
            f_full: (N_samples, N_x, N_v) - full distribution function at time t
            time_diffs: (N_samples,) - time difference between measurements
        """
        self.phi_sparse = torch.FloatTensor(phi_sparse)
        self.f_full = torch.FloatTensor(f_full)
        self.time_diffs = torch.FloatTensor(time_diffs)
        
    def __len__(self):
        return len(self.phi_sparse)
    
    def __getitem__(self, idx):
        # Flatten phi measurements and append time difference
        #TODO check that the indexing is correct here
        phi_t = self.phi_sparse[idx, 0, :]  # phi at time t
        phi_t_minus_1 = self.phi_sparse[idx, 1, :]  # phi at time t-1
        dt = self.time_diffs[idx:idx+1]  # time difference
        
        # Concatenate: [phi_t, phi_t-1, dt]
        #TODO is there a smarter way to do this?
        input_vector = torch.cat([phi_t, phi_t_minus_1, dt])
        
        # Flatten the output f
        target = self.f_full[idx].flatten()
        
        return input_vector, target


class MLP(nn.Module):
    def __init__(self, n_sparse_measurements, n_x_grid, n_v_grid, hidden_layers=[256, 512, 1024, 512]):
        """
        MLP to predict full distribution function from sparse potential measurements.
        
        Args:
            n_sparse_measurements: Number of sparse phi measurements per timestep
            n_x_grid: Number of spatial grid points (128)
            n_v_grid: Number of velocity grid points (64)
            hidden_layers: List of hidden layer sizes
        """
        super(MLP, self).__init__()
        
        # Input: sparse phi at t, sparse phi at t-1, and dt = 2*n_sparse + 1
        input_size = 2 * n_sparse_measurements + 1
        
        # Output: full f(x,v) distribution = n_x_grid * n_v_grid
        output_size = n_x_grid * n_v_grid
        
        self.n_x_grid = n_x_grid
        self.n_v_grid = n_v_grid
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Args:
            x: (batch, 2*n_sparse + 1) - sparse phi measurements and time diff
        Returns:
            f: (batch, n_x_grid, n_v_grid) - predicted distribution function
        """
        out = self.network(x)
        # Reshape to (batch, n_x, n_v)
        return out.view(-1, self.n_x_grid, self.n_v_grid)

def prepare_training_data(data_path, downsample_factor=10, train_split=0.8):
    """
    Load and prepare training data from simulation output.
    
    Args:
        data_path: Path to .npz file with simulation data
        downsample_factor: Factor to downsample spatial measurements
        train_split: Fraction of data to use for training
    
    Returns:
        train_dataset, val_dataset, n_sparse, n_x, n_v
    """
    # Load the data
    data = np.load(data_path, allow_pickle=True)
    
    times = data['times']  # (N_t,)
    electric = data['electric']  # (N_t, N_x)
    phase_density = data['phase_density']  # (N_t, N_x, N_v) - This is f!
    x_grid = data['x_grid']  # (N_x,)
    
    print(f"Loaded data:")
    print(f"  Times: {times.shape}")
    print(f"  Electric field: {electric.shape}")
    print(f"  Phase density (f): {phase_density.shape}")
    print(f"  X grid: {x_grid.shape}")
    
    # Compute electric potential from electric field
    phi = phi_from_E(electric, x_grid)  # (N_t, N_x)
    
    # Downsample phi spatially
    phi_sparse = phi[:, ::downsample_factor]  # (N_t, N_x/downsample_factor)
    n_sparse = phi_sparse.shape[1]
    n_x, n_v = phase_density.shape[1], phase_density.shape[2]
    
    print(f"  Sparse phi measurements: {n_sparse} per timestep")
    print(f"  Output grid: {n_x} x {n_v} = {n_x * n_v} values")
    
    # Create training samples: use pairs of consecutive timesteps
    # Each sample: [phi(t), phi(t-1), dt] -> f(t)
    N_samples = len(times) - 1
    
    phi_input = np.zeros((N_samples, 2, n_sparse))  # (N_samples, 2, n_sparse)
    f_output = np.zeros((N_samples, n_x, n_v))  # (N_samples, n_x, n_v)
    time_diffs = np.zeros(N_samples)
    
    for i in range(N_samples):
        phi_input[i, 0, :] = phi_sparse[i+1]  # phi at time t
        phi_input[i, 1, :] = phi_sparse[i]    # phi at time t-1
        f_output[i] = phase_density[i+1]      # f at time t (ground truth)
        time_diffs[i] = times[i+1] - times[i]  # dt
    
    # Split into train and validation
    split_idx = int(N_samples * train_split)
    
    train_dataset = VlasovDataset(
        phi_input[:split_idx], 
        f_output[:split_idx], 
        time_diffs[:split_idx]
    )
    val_dataset = VlasovDataset(
        phi_input[split_idx:], 
        f_output[split_idx:], 
        time_diffs[split_idx:]
    )
    
    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    return train_dataset, val_dataset, n_sparse, n_x, n_v


def train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cpu'):
    """
    Train the MLP model.
    
    Args:
        model: The MLP model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cpu' or 'cuda')
    """
    model = model.to(device)
    
    # Use MSE loss for regression (not CrossEntropyLoss which is for classification!)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                      factor=0.5, patience=5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs_flat = outputs.view(outputs.size(0), -1)  # Flatten for loss
            loss = criterion(outputs_flat, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                outputs_flat = outputs.view(outputs.size(0), -1)
                loss = criterion(outputs_flat, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses


if __name__ == "__main__":
    # Configuration
    # DATA_PATH = "/Users/david/270A_nudging/multiscale-nudging-main/case3_Vlasov_poisson_instability/simulation/data/mv_sim_seed0.npz"
    DATA_PATH = "/home/dschneidinger/270A_nudging/multiscale-nudging-main/case3_Vlasov_poisson_instability/simulation/data/mv_sim_seed0.npz"
    DOWNSAMPLE_FACTOR = 10  # Use 128/10 = ~13 sparse measurements
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Prepare data
    train_dataset, val_dataset, n_sparse, n_x, n_v = prepare_training_data(
        DATA_PATH, 
        downsample_factor=DOWNSAMPLE_FACTOR
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = MLP(
        n_sparse_measurements=n_sparse,
        n_x_grid=n_x,
        n_v_grid=n_v,
        hidden_layers=[256, 512, 1024, 512, 256]  # Deep network
    )
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    print("\nStarting training...\n")
    train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        device=device
    )
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training History')
    plt.grid(True)
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    print("\nSaved training history to training_history.png")
    
    # Save the trained model
    torch.save(model.state_dict(), 'vlasov_mlp_model.pth')
    print("Saved model to vlasov_mlp_model.pth")
    