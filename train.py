import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import argparse
from models.VAE import VAE
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os
import numpy as np

from utils.losses import loss_function  # Assuming you have defined this function

# Argument Parser
parser = argparse.ArgumentParser(description='Variational Autoencoder (VAE) for CIFAR-10')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--latent-dim', type=int, default=128, metavar='N',
                    help='dimensionality of the latent space (default: 128)')
parser.add_argument('--save-interval', type=int, default=5, metavar='N',
                    help='interval for saving model weights (default: 5)')
parser.add_argument('--validate-interval', type=int, default=1, metavar='N',
                    help='interval for validation (default: 1)')
parser.add_argument('--experiment-name', type=str, required=True, metavar='NAME',
                    help='name of the experiment')
parser.add_argument('--save-root', type=str, default='./vae_results', metavar='DIR',
                    help='root directory to save results, including images, weights, plots, and logs (default: ./vae_results)')
args = parser.parse_args()

# Define hyperparameters
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
latent_dim = args.latent_dim
save_interval = args.save_interval
validate_interval = args.validate_interval
experiment_name = args.experiment_name
save_root = os.path.join(args.save_root, experiment_name)

image_save_dir = os.path.join(save_root, 'resultimg')
weight_save_dir = os.path.join(save_root, 'weights')
plot_save_dir = os.path.join(save_root, 'plots')
log_file_path = os.path.join(save_root, f'{experiment_name}_training.log')
dataset_dir = os.path.join(save_root, 'dataset')

# Ensure directories exist, create if not
os.makedirs(image_save_dir, exist_ok=True)
os.makedirs(weight_save_dir, exist_ok=True)
os.makedirs(plot_save_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

# Log file setup
log_file = open(log_file_path, 'a', buffering=1)

# Log hyperparameters to file
log_file.write("Hyperparameters:\n")
log_file.write(f"Batch Size: {batch_size}\n")
log_file.write(f"Learning Rate: {learning_rate}\n")
log_file.write(f"Epochs: {epochs}\n")
log_file.write(f"Latent Dimension: {latent_dim}\n")
log_file.write(f"Save Interval: {save_interval}\n")
log_file.write(f"Validate Interval: {validate_interval}\n")
log_file.write(f"Experiment Name: {experiment_name}\n")
log_file.write(f"Image Save Directory: {image_save_dir}\n")
log_file.write(f"Weight Save Directory: {weight_save_dir}\n")
log_file.write(f"Plot Save Directory: {plot_save_dir}\n")
log_file.write(f"Dataset Root: {dataset_dir}\n\n")

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root=dataset_dir, train=True, transform=transform, download=True)
train_size = len(train_dataset)
train_set, val_set = random_split(train_dataset, [int(train_size * 0.8), int(train_size * 0.2)])

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Setting model and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=latent_dim).to(device)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Parallel GPU if available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)

train_losses = []
val_losses = []
train_epochs = []
val_epochs = []

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    # Wrap the train_loader with tqdm to show progress bar
    for batch_idx, (data, _) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    # Print average training loss for the epoch and log it
    avg_train_loss = train_loss / len(train_loader.dataset)
    log_file.write(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}\n")
    train_losses.append(avg_train_loss)
    train_epochs.append(epoch + 1)

    # Validation
    if (epoch + 1) % validate_interval == 0 or epoch == epochs - 1:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, (data_val, _) in enumerate(val_loader):
                data_val = data_val.to(device)
                recon_val, mu_val, logvar_val = model(data_val)
                val_loss += loss_function(recon_val, data_val, mu_val, logvar_val).item()

                # Save the first 5 pairs of original and reconstructed images
                if i < 5:
                    if i == 0:
                        validation_results = recon_val
                        original_images = data_val
                    else:
                        validation_results = torch.cat([validation_results, recon_val], dim=0)
                        original_images = torch.cat([original_images, data_val], dim=0)

            # Concatenate original and reconstructed images in a grid
            comparison = torch.cat([original_images[:5], validation_results[:5]])

            # Save the grid of images
            vutils.save_image(comparison, f"{image_save_dir}/vae_validation_comparison_epoch_{epoch + 1}.png", nrow=5,
                              normalize=True)

        avg_val_loss = val_loss / len(val_loader.dataset)
        log_file.write(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss:.4f}\n")
        val_losses.append(avg_val_loss)
        val_epochs.append(epoch + 1)

    # Save model weights
    if epoch % save_interval == 0:
        torch.save(model.state_dict(), f"{weight_save_dir}/vae_epoch_{epoch + 1}.pth")

    # Plotting training and validation losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_epochs, np.log(train_losses), label='Training Loss (log scale)', marker='o')
    plt.plot(val_epochs, np.log(val_losses), label='Validation Loss (log scale)', marker='s')
    plt.title('Training and Validation Losses (Log Scale)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{plot_save_dir}/training_validation_losses_logscale.png")

# Close the log file
log_file.close()
