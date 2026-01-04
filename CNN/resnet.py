import time
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision as tv

start_time = time.time()

device = "mps" if torch.mps.is_available() else "cpu"

# Training transforms
train_transforms = tv.transforms.Compose([
    tv.transforms.RandomCrop(32, padding=4),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761]),
])

# Testing transforms (deterministic)
test_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                            std=[0.2675, 0.2565, 0.2761]),
])

training_data = tv.datasets.CIFAR100(
    root="data",
    train=True,
    transform=train_transforms,
    download=True,
)

testing_data = tv.datasets.CIFAR100(
    root="data",
    train=False,
    transform=test_transforms,
    download=True,
)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=32, shuffle=True)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class RESNET18(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
        self.l1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False)
        self.l1_bn = nn.BatchNorm2d(64)
        self.l1_act = nn.ReLU(inplace=True)
        
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=1),
            BasicBlock(64, 64, stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1),
        )

        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, 100)
        
    def forward(self, X):
        
        X = self.l1(X)
        X = self.l1_bn(X)
        X = self.l1_act(X)
        
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        
        X = self.avg_pool(X)
        X = torch.flatten(X, 1)
        X = self.linear(X)
        
        return X

model = RESNET18()
model.to(device)
print(model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTrainable Parameters: {trainable_params}")

print("\nLet's Start Training:\n")
model.train()

epochs = 10
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs
)
loss = nn.CrossEntropyLoss()

for epoch in range(epochs):
    epoch_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        pred = model.forward(data)
        pred_loss = loss(pred, target)
        pred_loss.backward()
        epoch_loss += pred_loss
        optimizer.step()
        
        pred_correct = torch.argmax(pred, dim=1)
        correct += torch.sum(pred_correct == target)
        
    scheduler.step()
        
    avg_loss = epoch_loss / len(train_loader)
    correct = correct / len(training_data)
    
    print(f"Epoch: {epoch+1} \t|\t Loss: {epoch_loss:.2f} \t|\t Avg Loss: {avg_loss:.2f} \t|\t Accuracy: {correct:.2f}")
    
    
print("\nLet's start Evaluation:\n")
model.eval()

epoch = 0
epoch_loss = 0
correct = 0

with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data = data.to(device)
        target = target.to(device)
        
        pred = model(data)
        pred_loss = loss(pred, target)
        epoch_loss += pred_loss
        
        pred_correct = torch.argmax(pred, dim=1)
        correct += torch.sum(pred_correct == target)
    
    avg_loss = epoch_loss / len(test_loader)
    correct = correct / len(testing_data)

    print(f"Epoch: {0} \t|\t Loss: {epoch_loss:.2f} \t|\t Avg Loss: {avg_loss:.2f} \t|\t Accuracy: {correct:.2f}")

end_time = time.time()
print(f"\n\nExceution completed in: {(end_time - start_time):.4f}")