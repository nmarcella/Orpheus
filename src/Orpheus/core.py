import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import signal
import sys
import toml
from sklearn.decomposition import PCA
import os
import numpy as np


# Define a signal handler for graceful termination
def signal_handler(sig, frame):
    print("\nTerminating the program...")
    plt.close("all")  # Close all Matplotlib figures
    sys.exit(0)  # Exit the program


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals


class NeuralCA(nn.Module):
    def __init__(self, num_channels=16, grid_size=64):
        super(NeuralCA, self).__init__()
        self.num_channels = num_channels
        self.grid_size = grid_size
        
        # Define a convolutional layer to aggregate neighbor states
        self.perceive = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        
        # Define a small network to update cell states
        self.update_fn = nn.Sequential(
            nn.Linear(num_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_channels),
        )
    
    def forward(self, x):
        # Perceive neighbors
        neighbor_states = self.perceive(x)
        
        # Update state for each cell
        batch_size, channels, height, width = x.size()
        neighbor_states_flat = neighbor_states.view(batch_size, channels, -1).permute(0, 2, 1)
        updated_states_flat = self.update_fn(neighbor_states_flat)
        
        # Reshape back to grid form
        updated_states = updated_states_flat.permute(0, 2, 1).view(batch_size, channels, height, width)
        
        # Combine old and new states
        return x + updated_states


def initialize_grid(batch_size, num_channels, grid_size, device):
    # Random grid initialization
    grid = torch.zeros(batch_size, num_channels, grid_size, grid_size, device=device)
    center = grid_size // 2
    grid[:, 0, center, center] = 1.0  # Activate the center cell in the first channel
    return grid


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# Save checkpoint
def save_checkpoint(epoch, model, optimizer, grid, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, "model.pt")
    state_path = os.path.join(checkpoint_dir, "state.toml")

    # Save model and optimizer states
    torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()}, model_path)

    # Save training parameters and grid state
    state = {
        "epoch": epoch,
        "grid": grid.detach().cpu().numpy().tolist(),
    }
    with open(state_path, "w") as f:
        toml.dump(state, f)
    print(f"Checkpoint saved at epoch {epoch}.")


# Load checkpoint
def load_checkpoint(model, optimizer, device, checkpoint_dir="checkpoints"):
    model_path = os.path.join(checkpoint_dir, "model.pt")
    state_path = os.path.join(checkpoint_dir, "state.toml")

    if not os.path.exists(model_path) or not os.path.exists(state_path):
        print("No checkpoint found, starting from scratch.")
        return 0, None

    # Load model and optimizer states
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Load training parameters and grid state
    with open(state_path, "r") as f:
        state = toml.load(f)

    epoch = state["epoch"]
    grid = torch.tensor(state["grid"], device=device)

    print(f"Checkpoint loaded from epoch {epoch}.")
    return epoch, grid


# Visualization function
def visualize_pca(grid, epoch):
    grid_np = grid.detach().cpu().numpy()  # Convert grid to NumPy
    reshaped = grid_np[0].reshape(grid.shape[1], -1).T  # Shape: (height*width, channels)
    
    # Perform PCA
    pca = PCA(n_components=1)  # Reduce to 1 component
    pca_result = pca.fit_transform(reshaped)  # Shape: (height*width, 1)
    pca_grid = pca_result.reshape(grid.shape[2], grid.shape[3])  # Reshape to grid size
    
    plt.clf()
    plt.imshow(pca_grid, cmap="viridis")
    plt.colorbar()
    plt.title(f"PCA Visualization (Epoch {epoch})")
    plt.pause(0.1)

# Add Fourier Transform visualization function
def visualize_ft(grid, epoch):
    grid_np = grid.detach().cpu().numpy()  # Convert grid to NumPy
    ft_sum = np.zeros((grid.shape[2], grid.shape[3]))  # Initialize FT sum across channels

    for channel in range(grid.shape[1]):  # Iterate over all channels
        ft = np.fft.fft2(grid_np[0, channel, :, :])  # 2D Fourier Transform
        ft_magnitude = np.abs(np.fft.fftshift(ft))  # Shift zero frequency to center
        ft_sum += ft_magnitude  # Sum magnitudes across all channels

    plt.figure(figsize=(6, 6))
    plt.imshow(np.log1p(ft_sum), cmap="inferno")  # Log scale for better visualization
    plt.colorbar()
    plt.title(f"Fourier Transform Visualization (Epoch {epoch})")
    plt.pause(0.1)  # Pause to display the plot

# Visualization function for PCA and FT combined in one figure
def visualize_pca_and_ft(grid, epoch, pca_ax, ft_ax):
    grid_np = grid.detach().cpu().numpy()  # Convert grid to NumPy
    
    # PCA Visualization
    reshaped = grid_np[0].reshape(grid.shape[1], -1).T  # Shape: (height*width, channels)
    pca = PCA(n_components=1)  # Reduce to 1 component
    pca_result = pca.fit_transform(reshaped)  # Shape: (height*width, 1)
    pca_grid = pca_result.reshape(grid.shape[2], grid.shape[3])  # Reshape to grid size
    
    pca_ax.clear()
    pca_ax.imshow(pca_grid, cmap="viridis")
    pca_ax.set_title(f"PCA Visualization (Epoch {epoch})")
    pca_ax.axis("off")

    # Fourier Transform Visualization
    ft_sum = np.zeros((grid.shape[2], grid.shape[3]))  # Initialize FT sum across channels
    for channel in range(grid.shape[1]):  # Iterate over all channels
        ft = np.fft.fft2(grid_np[0, channel, :, :])  # 2D Fourier Transform
        ft_magnitude = np.abs(np.fft.fftshift(ft))  # Shift zero frequency to center
        ft_sum += ft_magnitude

    ft_ax.clear()
    ft_ax.imshow(np.log1p(ft_sum), cmap="inferno")  # Log scale for better visualization
    ft_ax.set_title(f"Fourier Transform (Epoch {epoch})")
    ft_ax.axis("off")

if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    num_channels = 16
    grid_size = 64
    batch_size = 64
    epochs = 100000
    steps_per_epoch = 16
    checkpoint_dir = "o6"
    visualize_step = 1
    save_checkpoint_step = 100

    # Initialize the NCA and optimizer
    nca = NeuralCA(num_channels=num_channels, grid_size=grid_size).to(device)
    nca.apply(init_weights)
    optimizer = optim.Adam(nca.parameters(), lr=1e-5)

    # Load checkpoint if exists
    start_epoch, grid = load_checkpoint(nca, optimizer, device, checkpoint_dir)

    # If no checkpoint, initialize grid
    if grid is None:
        grid = initialize_grid(batch_size, num_channels, grid_size, device=device)

    # Loss function
    def loss_fn(grid, prev_grid):
        activation = torch.mean(grid, dim=1)  # Average activation across all channels
        spread_loss = -torch.mean(activation)  # Encourage activation spread
        regularization = 1e-4 * torch.sum(grid ** 2)  # Regularization
        change_loss = torch.mean((grid - prev_grid) ** 2)
        return spread_loss + regularization + 1e-2 * change_loss

    # Persistent figure and axes for visualization
    plt.ion()  # Turn on interactive mode
    fig, (pca_ax, ft_ax) = plt.subplots(1, 2, figsize=(12, 6))  # Single figure with two subplots

    # Training loop
    try:
        for epoch in range(start_epoch, epochs):  # Training loop
            optimizer.zero_grad()

            prev_grid = grid.detach().clone()

            # Simulate multiple steps
            noise_level = 0.01
            for _ in range(steps_per_epoch):
                grid = nca(grid)
                grid = grid + noise_level * torch.randn_like(grid, device=device)  # Add random noise
                grid = torch.clamp(grid, min=-1, max=1)  # Keep values bounded
                grid = grid.detach()
                grid.requires_grad_()

            # Compute loss
            loss = loss_fn(grid, prev_grid)
            loss.backward()
            optimizer.step()

            # Save checkpoint every 100 epochs
            if epoch % save_checkpoint_step == 0:
                save_checkpoint(epoch, nca, optimizer, grid, checkpoint_dir)

            # Visualization
            if epoch % visualize_step == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                #visualize_pca(grid, epoch)

                # # Show PCA and FT visualizations
                # plt.figure(figsize=(12, 6))  # Create a shared figure for both visualizations
                
                # # Subplot for PCA
                # plt.subplot(1, 2, 1)
                # visualize_pca(grid, epoch)
                
                # # Subplot for Fourier Transform
                # plt.subplot(1, 2, 2)
                # visualize_ft(grid, epoch)

                # plot both
            # Visualization
            if epoch % visualize_step == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
                visualize_pca_and_ft(grid, epoch, pca_ax, ft_ax)
                plt.pause(0.1)  # Pause to update the plot


    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        plt.ioff()
        #visualize_pca(grid.cpu(), epoch="Final")
        #visualize_ft(grid.cpu(), epoch="Final")
        plt.show()





# import torch
# import torch.nn as nn
# import torch.optim as optim
# import matplotlib.pyplot as plt
# import signal
# import sys
# from sklearn.decomposition import PCA


# # Define a signal handler for graceful termination
# def signal_handler(sig, frame):
#     print("\nTerminating the program...")
#     plt.close("all")  # Close all Matplotlib figures
#     sys.exit(0)  # Exit the program


# # Register the signal handler
# signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
# signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals


# class NeuralCA(nn.Module):
#     def __init__(self, num_channels=16, grid_size=64):
#         super(NeuralCA, self).__init__()
#         self.num_channels = num_channels
#         self.grid_size = grid_size
        
#         # Define a convolutional layer to aggregate neighbor states
#         self.perceive = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        
#         # Define a small network to update cell states
#         self.update_fn = nn.Sequential(
#             nn.Linear(num_channels, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_channels),
#         )
    
#     def forward(self, x):
#         # Perceive neighbors
#         neighbor_states = self.perceive(x)
        
#         # Update state for each cell
#         batch_size, channels, height, width = x.size()
#         neighbor_states_flat = neighbor_states.view(batch_size, channels, -1).permute(0, 2, 1)
#         updated_states_flat = self.update_fn(neighbor_states_flat)
        
#         # Reshape back to grid form
#         updated_states = updated_states_flat.permute(0, 2, 1).view(batch_size, channels, height, width)
        
#         # Combine old and new states
#         return x + updated_states


# def initialize_grid(batch_size, num_channels, grid_size, device):
#     # Random grid initialization
#     grid = torch.zeros(batch_size, num_channels, grid_size, grid_size, device=device)
#     center = grid_size // 2
#     grid[:, 0, center, center] = 1.0  # Activate the center cell in the first channel
#     return grid


# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)


# # Visualization function
# def visualize_grid(grid, epoch):
#     plt.clf()  # Clear the current figure
#     activation = grid[0, 0].detach().cpu().numpy()  # Visualize first batch, first channel
#     plt.imshow(activation, cmap="viridis")
#     plt.colorbar()
#     plt.title(f"Epoch {epoch}")
#     plt.pause(0.1)  # Pause to update the figure


# def visualize_pca(grid, epoch):
#     grid_np = grid.detach().cpu().numpy()  # Convert grid to NumPy
#     reshaped = grid_np[0].reshape(grid.shape[1], -1).T  # Shape: (height*width, channels)
    
#     # Perform PCA
#     pca = PCA(n_components=1)  # Reduce to 1 component
#     pca_result = pca.fit_transform(reshaped)  # Shape: (height*width, 1)
#     pca_grid = pca_result.reshape(grid.shape[2], grid.shape[3])  # Reshape to grid size
    
#     plt.clf()
#     plt.imshow(pca_grid, cmap="viridis")
#     plt.colorbar()
#     plt.title(f"PCA Visualization (Epoch {epoch})")
#     plt.pause(0.1)



# if __name__ == "__main__":
#     # Configuration
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     num_channels = 16 # states per cell
#     grid_size = 64
#     batch_size = 8
#     steps_per_epoch = 16 # might increase complexity

#     # Initialize the NCA and optimizer
#     nca = NeuralCA(num_channels=num_channels, grid_size=grid_size).to(device)
#     nca.apply(init_weights)
#     optimizer = optim.Adam(nca.parameters(), lr=1e-5)

#     # Initialize the grid
#     grid = initialize_grid(batch_size, num_channels, grid_size, device=device)

#     # Loss function
#     def loss_fn(grid, prev_grid):
#         #activation = grid[:, 0, :, :]  # Use the first channel as "activation" meaning this is the primairy spread behavior.
#         activation = torch.mean(grid, dim=1)  # Average activation across all channels
#         spread_loss = -torch.mean(activation)  # Encourage activation spread
#         regularization = 1e-4 * torch.sum(grid ** 2)  # Regularization

#         # New: Penalize similarity with the previous state
#         change_loss = torch.mean((grid - prev_grid) ** 2)

#         return spread_loss + regularization + 1e-2 * change_loss

#     # Set up real-time visualization
#     plt.ion()
#     plt.figure(figsize=(6, 6))

#     # Training loop
#     try:
#         for epoch in range(10000):  # Training loop
#             optimizer.zero_grad()

#             prev_grid = grid.detach().clone()  # Save a detached copy of the current grid

#             # Simulate multiple steps
#             noise_level = 0.01
#             #noise = torch.randn_like(grid, device=device) * noise_level

#             for _ in range(steps_per_epoch):
#                 grid = nca(grid)
#                 grid = grid + noise_level * torch.randn_like(grid, device=device)  # Add random noise
#                 grid = torch.clamp(grid, min=-1, max=1)  # Keep values bounded
#                 grid = grid.detach()
#                 grid.requires_grad_()

#             # Compute loss
#             loss = loss_fn(grid, prev_grid)  # Pass previous grid to loss
#             loss.backward()
#             #torch.nn.utils.clip_grad_norm_(nca.parameters(), max_norm=1.0)  # Clip gradients to prevent explosion
#             optimizer.step()

#             # Display progress
#             if epoch % 10 == 0:
#                 print(f"Epoch {epoch}, Loss: {loss.item()}")
            
#             # Real-time visualization
#             if epoch % 1 == 0:
#                 #visualize_grid(grid, epoch)
#                 visualize_pca(grid, epoch)

#     except KeyboardInterrupt:
#         # Catch Ctrl+C and clean up
#         print("\nTraining interrupted by user.")
#     finally:
#         plt.ioff()  # Turn off interactive mode
#         visualize_grid(grid.cpu(), epoch="Final")  # Show the final grid state
#         plt.show()  # Keep the final figure open
