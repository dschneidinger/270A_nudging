# Train a Network to find a distribution function f, from a given potential phi, measured sparsely in space.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
"""
Vlasov equation : \partial_t f + v \cdot \nabla_x f + F \cdot \nabla_v f = 0
For now we will just consider the electrostatic case, where F = q E = -\nabla_x * \phi * q

Full distribution function f(x1,v1,t) is output from the numerical solver.
In the future this could be from osiris, but for now it will be from the numerical solver Hayden's postdoc wrote
"""

def phi_from_f(f: np.ndarray)-> np.ndarray:
    """
    f : distribution function, of shape (num_x, num_v, num_t)
    phi : potential, of shape (num_x, num_t)
    """
    assert np.shape(f) == np.shape(phi[...])
    return phi

# Source: https://www.codegenes.net/blog/how-to-do-mlp-pytorch/
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # For the input layer, have n measurements of phi at positions x1, x2, ..., xn
        # In order to correctly get the dynamics, we need to also include the potential at t-1
        # I presume that it also needs information about the difference in time between each measurement

        super(MLP, self).__init__() # Instantiate 
        self.fc1 = nn.Linear(input_size, hidden_size) #Fully connected layer 1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Initialize the MLP
input_size = 10
hidden_size = 20
output_size = 2
mlp = MLP(input_size, hidden_size, output_size)

# Load in the total data set, f
f = np.load("distribution_function.npy") #TODO fix this
# Assume that f is of shape num_x x num_v x num_t
# We can analytically derive the potential phi from f
phi = phi_from_f(f)
# For our input into the MLP, we downsample phi at n positions in space, but keep full time resolution.
# However, we will need to play with the number of different time steps we can give the MLP. It requires at least 2

def get_training_data(phi, n_skip, n_times):
    # Downsample phi at n positions in space
    num_x, num_t = phi.shape
    downsampled_phi = phi[::n_skip, :]
    # Now we have a downsampled phi of shape n_downsample x num_t
    # We need to create training data of shape (num_samples, input_size)
    # where input_size = n_downsample * n_times
    num_samples = num_t - n_times + 1
    training_data = np.zeros((num_samples, input_size))
    for i in range(num_samples):
        training_data[i] = downsampled_phi[:, i:i+n_times].flatten()
    return training_data


output = mlp(torch.from_numpy(f).float()) #TODO

 
# Generate some dummy data for training
train_input = torch.randn(100, input_size)
train_labels = torch.randint(0, output_size, (100,))
 
# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mlp.parameters(), lr=0.01)
 
# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = mlp(train_input)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mlp.to(device)
train_input = train_input.to(device)
train_labels = train_labels.to(device)