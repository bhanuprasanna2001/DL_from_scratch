import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. LOAD FASHION MNIST
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

# 2. MODEL (Conv Autoencoder)
class DenoisingAE(nn.Module):
    def __init__(self):
        super(DenoisingAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 3. TRAIN WITH NOISE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAE().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def add_noise(img, noise_factor=0.3):
    noise = torch.randn_like(img) * noise_factor
    return torch.clip(img + noise, 0., 1.)

print("Training on Fashion-MNIST...")
for epoch in range(5):
    for img, _ in train_loader:
        img = img.to(device)
        noisy_img = add_noise(img)
        
        output = model(noisy_img)
        loss = criterion(output, img) # Compare Output vs CLEAN Image
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 4. VISUALIZE RESULTS
# Get a test batch
clean_imgs, _ = next(iter(test_loader))
clean_imgs = clean_imgs.to(device)
noisy_imgs = add_noise(clean_imgs)

model.eval()
with torch.no_grad():
    denoised_imgs = model(noisy_imgs)

# Plot: Top=Noisy, Bottom=Cleaned
plt.figure(figsize=(10, 4))
for i in range(5):
    # Plot Noisy
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(noisy_imgs[i].cpu().squeeze(), cmap='gray')
    plt.title("Input (Noisy)")
    plt.axis("off")
    
    # Plot Denoised
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.imshow(denoised_imgs[i].cpu().squeeze(), cmap='gray')
    plt.title("Output (Clean)")
    plt.axis("off")
plt.show()