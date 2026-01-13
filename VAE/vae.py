"""
Variational Autoencoder (VAE) implementation on MNIST.

A VAE learns a probabilistic latent representation by minimizing:
    L = E[log p(x|z)] - β·KL(q(z|x) || p(z))
    
The first term reconstructs images, the second regularizes the latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.distributions import Normal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path


class VAE(nn.Module):
    """
    Variational Autoencoder with fully connected encoder/decoder.
    
    Architecture:
        Encoder: 784 → 512 → 256 → 2*latent_dim (μ, log σ²)
        Decoder: latent_dim → 256 → 512 → 784
    """
    
    def __init__(self, latent_dim=20, beta=1.0):
        """
        Args:
            latent_dim: Dimensionality of latent space
            beta: Weighting factor for KL divergence (β-VAE)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: maps x → (μ, log σ²)
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        
        # Decoder: maps z → x
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid(),  # Pixel values in [0, 1]
        )
    
    def encode(self, x):
        """Encode image to latent space: q(z|x) = N(μ, σ²I)"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample z ~ N(μ, σ²I).
        
        z = μ + σ ⊙ ε,  where ε ~ N(0, I)
        
        This allows gradients to flow through the sampling.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def decode(self, z):
        """Decode latent code to image: p(x|z)"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass: encode, sample, decode"""
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar):
        """
        VAE loss: ELBO (Evidence Lower Bound)
        
        L = Σ log p(x|z) - β·KL(q(z|x) || p(z))
        
        where:
        - Reconstruction: Binary Cross Entropy (summed per batch)
        - KL: 0.5 * Σ(μ² + σ² - log σ² - 1) (averaged per batch)
        """
        # Reconstruction loss: BCE summed across pixels, averaged across batch
        # Using reduction='none' then mean() to ensure correct scaling
        recon_loss = F.binary_cross_entropy(
            recon_x, x.view(-1, 784), reduction='none'
        ).sum(dim=1).mean()
        
        # KL divergence: averaged per latent dimension and batch
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss
    
    def sample(self, n_samples=16):
        """Sample new images by drawing z ~ N(0, I) and decoding."""
        z = torch.randn(n_samples, self.latent_dim)
        with torch.no_grad():
            samples = self.decode(z)
        return samples


def load_mnist(batch_size=128):
    """Load MNIST dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, train_dataset, test_dataset


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for x, _ in loader:
        x = x.to(device)
        optimizer.zero_grad()
        
        recon_x, mu, logvar, z = model(x)
        loss, recon_loss, kl_loss = model.loss_function(recon_x, x, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    
    for x, _ in loader:
        x = x.to(device)
        recon_x, mu, logvar, z = model(x)
        loss, _, _ = model.loss_function(recon_x, x, mu, logvar)
        total_loss += loss.item()
    
    return total_loss / len(loader)


def train_vae(model, train_loader, test_loader, epochs=30, lr=1e-3, device='cpu'):
    """Train VAE"""
    optimizer = Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        test_loss = evaluate(model, test_loader, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    
    print(f"Epoch {epochs:3d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
    return train_losses, test_losses


def main():
    """Main training pipeline"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    train_loader, test_loader, train_dataset, test_dataset = load_mnist(batch_size=128)
    print(f"Loaded MNIST: {len(train_dataset)} train, {len(test_dataset)} test images\n")
    
    # Create output directory
    output_dir = Path("./outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Train VAEs with different β values
    beta_values = [0, 1, 4]
    models = {}
    
    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"Training VAE with β = {beta}")
        print(f"{'='*60}")
        
        model = VAE(latent_dim=4, beta=beta).to(device)
        train_losses, test_losses = train_vae(
            model, train_loader, test_loader, epochs=30, lr=1e-3, device=device
        )
        
        models[beta] = model
        
        # Save model
        torch.save(model.state_dict(), output_dir / f"vae_beta_{beta}.pth")
    
    print(f"\n{'='*60}")
    print("Training complete! Models saved to ./outputs/")
    print(f"{'='*60}\n")
    
    return models, test_loader, test_dataset


if __name__ == "__main__":
    models, test_loader, test_dataset = main()
