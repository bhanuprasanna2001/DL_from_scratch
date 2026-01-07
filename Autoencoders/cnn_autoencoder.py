# Autoencoder has mainly 3 components:
# 1. Encoder
# 2. Bottleneck
# 3. Decoder

# What we are trying to do with Autoencoder is that we reducing the dimensions required
# represent the data at hand.

# We have this Dimensionality Reduction Techniques right:
# 1. PCA
# 2. T-SNE
# 3. U-Map
# this are the ones that I am familiar with.

# The PCA is the only Linear method, I have implemented a from scratch version in ML_from_scratch repo.

# The rest, T-SNE, U-Map, and the Autoencoder are all Non-Linear Dimensionality Reduction Techniques.

# AEs: Reconstruct the input data from the compressed representation.
# PCA: Maximize variance captured by components.
# t-SNE/UMAP: Preserve neighborhood similarities or distances in the lower-dimensional space.

# PCA is fast; UMAP is efficient; t-SNE can be slow; AEs vary but can be slow to train.

# PCA for linear patterns, t-SNE/UMAP for visualizing clusters/manifolds, AEs for learning powerful feature representations for various downstream tasks. 

# Differences between Autoencoder and other algorithms:

# Autoencoder (AE): A neural network (encoder-decoder) that learns a non-linear compression (latent space) by minimizing reconstruction error, 
# making it highly flexible for complex data but computationally intensive to train.

# PCA (Principal Component Analysis): A linear method finding orthogonal axes (principal components) that capture the most variance, excellent 
# for large datasets and interpretable results but misses non-linear patterns.

# t-SNE (t-Distributed Stochastic Neighbor Embedding): A non-linear method focused on preserving local neighborhood structures, ideal for 
# visualizing clusters but computationally expensive (O(N²)) and less effective at preserving global structure.

# UMAP (Uniform Manifold Approximation and Projection): A non-linear method that balances preserving local and global structures, generally 
# faster and more scalable than t-SNE, making it great for clustering and visualization. 


# For this version, we can do image reconstruction through Linear layers in pytorch, by flattening the image,
# but we would lose the spatial information, the network doesn't know if the pixel 1 is physically next to pixel 28 in the grid.

# So, here I will be doing with CNN.

# I thought I will be using the same CNN layers in the decoder, it seems there is this CNN Transpose which reverses
# the spatial information (or upsamples the dimensions).

# Standard convolution typically performs downsampling using strides greater than 1 or pooling layers.

# The formulae for Standard CNN output shape is:
    
# h_width = ((height + 2 * padding - kernel) // stride) + 1
# w_width = ((width + 2 * padding - kernel) // stride) + 1

# The formulae for Transposed CNN output shape is:

# h_width = (height - 1) * stride + kernel  - 2 * padding
# w_width = (width - 1) * stride + kernel  - 2 * padding

import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

# 1. Data Preparation
# We do NOT flatten here. We keep it as a Tensor (1, 28, 28)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

# 2. Define the Model
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # --- ENCODER ---
        # Compresses the image
        self.encoder = nn.Sequential(
            # Input: 1 x 28 x 28
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # Output: 16 x 14 x 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # Output: 32 x 7 x 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 7) # Output: 64 x 1 x 1 (This is the latent representation)
        )
        
        # --- DECODER ---
        # Reconstructs the image
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7), # Output: 32 x 7 x 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # Output: 16 x 14 x 14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), # Output: 1 x 28 x 28
            nn.Sigmoid() # Output pixels between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. Training Setup
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model = ConvAutoencoder().to(device)
criterion = nn.MSELoss() # Compare Pixel vs Pixel
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training Loop
num_epochs = 5
print("Starting Training...")
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data # We don't need labels (this is unsupervised)
        img = img.to(device)
        
        # Forward pass
        output = model(img)
        loss = criterion(output, img) # Calculate error between Input and Output
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Visualization
# Let's check the results on test data
model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        
        # Plot the first 5 input vs output
        plt.figure(figsize=(10, 4))
        for i in range(5):
            # Original
            ax = plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].cpu().reshape(28, 28), cmap='gray')
            plt.title("Original")
            plt.axis("off")
            
            # Reconstructed
            ax = plt.subplot(2, 5, i + 1 + 5)
            plt.imshow(outputs[i].cpu().reshape(28, 28), cmap='gray')
            plt.title("Recon")
            plt.axis("off")
        plt.show()
        break

# 1. Helper function to get the "Bottleneck" output
# We need to hook into the model and stop halfway
def get_latent_vectors(model, loader, device):
    model.eval()
    latent_vectors = []
    labels = []
    
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            # Pass through ENCODER only
            # Note: We need to access the encoder part of the model we defined earlier
            encoded_output = model.encoder(images) 
            
            # Flatten the output: (Batch_Size, 64, 1, 1) -> (Batch_Size, 64)
            encoded_output = encoded_output.view(encoded_output.size(0), -1)
            
            latent_vectors.append(encoded_output.cpu().numpy())
            labels.append(targets.numpy())
            
            # Let's just do 1000 images to save time/memory for this demo
            if len(latent_vectors) * loader.batch_size > 1000:
                break
                
    return np.concatenate(latent_vectors), np.concatenate(labels)

# 2. Extract the Data
print("Extracting latent vectors...")
latent_vecs, targets = get_latent_vectors(model, test_loader, device)

# 3. Visualize 1: The "Barcode"
# Save barcode for each unique digit in a folder
# Create output directory
output_dir = "./barcode_plots"
os.makedirs(output_dir, exist_ok=True)

# Get unique targets and save one barcode per digit
unique_targets = np.unique(targets)

for digit in unique_targets:
    # Find first occurrence of this digit
    idx = np.where(targets == digit)[0][0]
    
    plt.figure(figsize=(10, 4))
    plt.bar(range(64), latent_vecs[idx])
    plt.title(f"The 'DNA' of the Digit: {digit}")
    plt.xlabel("Neuron Index (0-63)")
    plt.ylabel("Activation Value")
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, f"barcode_digit_{digit}.png"), dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Saved barcode for digit {digit}")

print(f"All barcodes saved to '{output_dir}'")

# This prevents the thread contention that causes the SegFault on M1/M2 Macs
os.environ['OMP_NUM_THREADS'] = '1'

# 4. Visualize 2: The "Cluster Map" (t-SNE)
print("Running t-SNE (this might take a moment)...")
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_vecs)

# Plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=targets, cmap='jet', alpha=0.7)
plt.colorbar(scatter)
plt.title("The Autoencoder's 'Brain' (t-SNE of Bottleneck)")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.show()