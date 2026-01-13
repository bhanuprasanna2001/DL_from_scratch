"""
VAE Visualization and Analysis script.

Demonstrates:
1. Reconstructions: Original → Reconstruction for 20 random test images
2. Sampling: Generate new digits from z ~ N(0, I)
3. Latent space map: 2D latent space colored by digit label (shows clustering)
4. Interpolation: Smooth morphing between two images in latent space
5. β-VAE comparison: Visual impact of β on reconstructions and samples
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

from vae import VAE, load_mnist


def plot_reconstructions(model, test_loader, device, beta=1, n_images=20):
    """
    Visualize: Original image → Reconstruction
    
    Shows how well the VAE reconstructs test images.
    Lower reconstruction error = better latent representation.
    """
    model.eval()
    
    # Get first batch
    x, labels = next(iter(test_loader))
    x = x[:n_images].to(device)
    
    with torch.no_grad():
        recon_x, _, _, _ = model(x)
    
    x = x.cpu().numpy().reshape(-1, 28, 28)
    recon_x = recon_x.cpu().numpy().reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(2, n_images // 2, figsize=(20, 5))
    fig.suptitle(f"Reconstructions (β = {beta}): Top=Original, Bottom=Reconstruction", fontsize=14, fontweight='bold')
    
    for i in range(n_images // 2):
        # Original
        axes[0, i].imshow(x[i], cmap='gray')
        axes[0, i].set_title(f"{i}")
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(recon_x[i], cmap='gray')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_samples(model, device, beta=1, n_samples=16):
    """
    Generate new images by sampling z ~ N(0, I) and decoding.
    
    These are completely new images the model has never seen—
    demonstrates that the latent space is meaningful and continuous.
    """
    model.eval()
    
    with torch.no_grad():
        samples = model.sample(n_samples).cpu().numpy().reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle(f"Samples from VAE (β = {beta}): z ~ N(0, I)", fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i], cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_latent_space(model, test_loader, device, beta=1):
    """
    2D latent space visualization (only meaningful if latent_dim=2).
    
    Each point is an encoded test image, colored by digit label.
    Well-trained VAEs show clear clustering by digit—
    this is the learned structure in latent space.
    """
    if model.latent_dim != 2:
        print(f"⚠️  Latent space visualization only works for latent_dim=2. Current: {model.latent_dim}")
        return None
    
    model.eval()
    
    encoded_points = []
    labels_list = []
    
    with torch.no_grad():
        for x, labels in test_loader:
            x = x.to(device)
            mu, _ = model.encode(x.view(-1, 784))
            encoded_points.append(mu.cpu().numpy())
            labels_list.append(labels.numpy())
    
    encoded_points = np.concatenate(encoded_points, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for digit in range(10):
        mask = labels_list == digit
        ax.scatter(
            encoded_points[mask, 0],
            encoded_points[mask, 1],
            c=[colors[digit]],
            label=str(digit),
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel("z₁ (latent dim 0)", fontsize=12, fontweight='bold')
    ax.set_ylabel("z₂ (latent dim 1)", fontsize=12, fontweight='bold')
    ax.set_title(f"2D Latent Space (β = {beta}): Colored by Digit Label", fontsize=14, fontweight='bold')
    ax.legend(title="Digit", loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_interpolation(model, test_dataset, device, beta=1, n_steps=15):
    """
    Interpolation in latent space: z(t) = (1-t)·z_a + t·z_b
    
    Pick two random images, encode them to z_a and z_b,
    then linearly interpolate in latent space.
    
    Shows smooth morphing between digits—indicates the VAE
    learned a meaningful, continuous latent representation.
    """
    model.eval()
    
    # Pick two random test images
    idx_a, idx_b = np.random.choice(len(test_dataset), 2, replace=False)
    x_a, label_a = test_dataset[idx_a]
    x_b, label_b = test_dataset[idx_b]
    
    x_a = x_a.unsqueeze(0).to(device)
    x_b = x_b.unsqueeze(0).to(device)
    
    with torch.no_grad():
        z_a, _ = model.encode(x_a.view(-1, 784))
        z_b, _ = model.encode(x_b.view(-1, 784))
    
    # Interpolate in latent space
    t_values = np.linspace(0, 1, n_steps)
    interpolated_images = []
    
    with torch.no_grad():
        for t in t_values:
            z_t = (1 - t) * z_a + t * z_b
            x_t = model.decode(z_t)
            interpolated_images.append(x_t.cpu().numpy())
    
    interpolated_images = np.array(interpolated_images).reshape(-1, 28, 28)
    
    fig, axes = plt.subplots(1, n_steps, figsize=(20, 3))
    fig.suptitle(f"Interpolation (β = {beta}): Morphing from digit {label_a} to {label_b}", 
                 fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes):
        ax.imshow(interpolated_images[i], cmap='gray')
        ax.set_title(f"t={t_values[i]:.2f}", fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_beta_comparison(models, test_loader, device):
    """
    Compare samples across different β values.
    
    β = 0: No KL regularization → poor generalization, mode collapse
    β = 1: Standard VAE → balanced reconstruction and regularization
    β = 4: Strong KL → focuses on learning structure → better interpolation
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 4, figure=fig)
    
    beta_values = sorted(models.keys())
    
    for row, beta in enumerate(beta_values):
        model = models[beta].to(device)
        model.eval()
        
        # Row 1: Samples
        with torch.no_grad():
            samples = model.sample(4).cpu().numpy().reshape(-1, 28, 28)
        
        for col in range(4):
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(samples[col], cmap='gray')
            if col == 0:
                ax.set_ylabel(f"β = {beta}\n(Samples)", fontweight='bold', fontsize=11)
            ax.axis('off')
    
    fig.suptitle("Impact of β on Generated Samples\nβ=0: Low KL | β=1: Balanced | β=4: High KL (Structure)", 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig


def visualize_all(models, test_loader, test_dataset, device):
    """Generate all visualizations"""
    output_dir = Path("./outputs/visualizations")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60 + "\n")
    
    # Pick one β value for detailed visualizations
    beta_detailed = 1
    model = models[beta_detailed].to(device)
    
    # 1. Reconstructions
    print("📊 Generating reconstruction visualizations...")
    fig = plot_reconstructions(model, test_loader, device, beta=beta_detailed, n_images=20)
    plt.savefig(output_dir / f"01_reconstructions_beta_{beta_detailed}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Samples
    print("🎲 Generating sample visualizations...")
    fig = plot_samples(model, device, beta=beta_detailed, n_samples=16)
    plt.savefig(output_dir / f"02_samples_beta_{beta_detailed}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Latent space (only if latent_dim == 2)
    if model.latent_dim == 2:
        print("📍 Generating 2D latent space visualization...")
        fig = plot_latent_space(model, test_loader, device, beta=beta_detailed)
        if fig:
            plt.savefig(output_dir / f"03_latent_space_beta_{beta_detailed}.png", dpi=150, bbox_inches='tight')
            plt.close()
    else:
        print(f"⏭️  Skipping 2D latent space visualization (latent_dim={model.latent_dim}, requires 2)")
    
    # 4. Interpolation
    print("🔄 Generating interpolation visualization...")
    fig = plot_interpolation(model, test_dataset, device, beta=beta_detailed, n_steps=15)
    plt.savefig(output_dir / f"04_interpolation_beta_{beta_detailed}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. β comparison
    print("⚖️  Generating β comparison visualization...")
    fig = plot_beta_comparison(models, test_loader, device)
    plt.savefig(output_dir / f"05_beta_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ All visualizations saved to {output_dir}\n")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, test_loader, train_dataset, test_dataset = load_mnist(batch_size=128)
    
    # Load trained models
    output_dir = Path("./outputs")
    models = {}
    
    for beta in [0, 1, 4]:
        model = VAE(latent_dim=20, beta=beta)
        model_path = output_dir / f"vae_beta_{beta}.pth"
        
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=device))
            models[beta] = model
        else:
            print(f"⚠️  Model for β={beta} not found. Run vae.py first to train models.")
    
    if models:
        visualize_all(models, test_loader, test_dataset, device)
    else:
        print("❌ No trained models found. Please run vae.py first.")
