# From now onwards, I will be only working with torch. The scratch implementations are done :)

# Only one concept that is at the core is:
# The calculation of what is sent to the next layer.

# The usage of pooling layer just after cnn. Pooling is much faster than batch normalization.
# Pooling reduces the computational cost by shrinking spatial dimensions, making subsequent
# layers faster. BatchNorm adds overhead but stabilizes training and can lead to faster 
# convergence overall.

# Pooling is a downsamples the spatial dimensions.

# Because it is not strict to just use the pooling layer after cnn, because the usage of pooling
# layer, we lose the spatial information or the spatial depth. Like for semantic segmentation, or 
# complex Computer Vision tasks, it is highly important to preserve the spatial information.

# But general classification task it is fine we use pooling, but it is important to always consider 
# what data are we dealing with. Here I am using MNIST which is just a black and white images.

# Batch normalization normalizes feature distribution using the mean and variance. BN is not just
# used after conv but can also be used after linear, because it is just a normalization technique.

# For example:
# INPUT: (batch=1, channels=32, height=28, width=28)

# Pooling: 
# pool = nn.MaxPool2d(2,2)
# output = pool(input)
# Output: (1, 32, 14, 14) -> Half the spatial size

# How did we calculate this?
# here padding = 0, stride = 2, kernel = 2
# h_width = ((height + 2 * padding - kernel) // stride) + 1 = ((28 + 0 - 2) // 2) + 1 = 14
# w_width = ((width + 2 * padding - kernel) // stride) + 1 = ((28 + 0 - 2) // 2) + 1 = 14

# Batch Normalization
# bn = nn.BatchNorm2d(32)
# output = bn(input)
# Output: (1, 32, 28, 28) -> Same shape

# We lose pixel location when we do pooling, while the BN preserves it.

# Standard pattern that we observe is : Conv -> BatchNorm -> ReLU -> MaxPool (repeat 2 - 3 times) -> Flatten


import torch
from torch import nn
import torchvision as tv

device = "mps" if torch.mps.is_available() else "cpu"

training_data = tv.datasets.MNIST(
    root="data",
    train=True,
    transform=tv.transforms.ToTensor(),
    download=True,
)

testing_data = tv.datasets.MNIST(
    root="data",
    train=False,
    transform=tv.transforms.ToTensor(),
    download=True,
)


train_data = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
test_data = torch.utils.data.DataLoader(testing_data, batch_size=32, shuffle=True)

print(len(training_data.data), len(train_data))
print(len(testing_data.data), len(test_data), "\n")


class cnn(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.c1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.ch1 = nn.ReLU()
        self.flat = nn.Flatten()
        self.z1 = nn.LazyLinear(128)
        self.h1 = nn.ReLU()
        self.z2 = nn.Linear(128, 10)
        
    def forward(self, X):
        X = self.c1(X)
        X = self.b1(X)
        X = self.ch1(X)
        X = self.flat(X)
        X = self.z1(X)
        X = self.h1(X)
        X = self.z2(X)
        
        return X
    

model = cnn()
print(model, "\n")
model.to(device)

model.train()
optimizer = torch.optim.SGD(model.parameters())
loss = torch.nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_data):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        y_pred = model.forward(data)
        batch_loss = loss(y_pred, target)
        epoch_loss+=batch_loss
        batch_loss.backward()
        optimizer.step()
        
        y_pred_correct = torch.argmax(y_pred, dim=1)
        correct += torch.sum(y_pred_correct == target)

    correct = correct / len(training_data.data)
    avg_loss = epoch_loss / len(train_data)
    print(f"Epoch: {epoch+1}, AvgLoss: {avg_loss:.4f}, Total Loss: {epoch_loss:.4f}, Accuracy: {correct:.4f}")

print()

model.eval()
with torch.no_grad():
    epoch_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(test_data):
        data = data.to(device)
        target = target.to(device)
        
        y_pred = model.forward(data)
        batch_loss = loss(y_pred, target)
        epoch_loss+=batch_loss
        
        y_pred_correct = torch.argmax(y_pred, dim=1)
        correct += torch.sum(y_pred_correct == target)
    
    correct = correct / len(testing_data.data)
    avg_loss = epoch_loss / len(test_data)
    
    print(f"AvgLoss: {avg_loss:.4f}, Total Loss: {epoch_loss:.4f}, Accuracy: {correct:.4f}")