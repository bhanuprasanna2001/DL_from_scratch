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

train_loader = torch.utils.data.DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=32, shuffle=True)


class LeNet5_S(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.seq_model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
        
    
    def forward(self, X):
        X = self.seq_model(X)
        return X
    
    
class LeNet5_R(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.seq_model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
        
    
    def forward(self, X):
        X = self.seq_model(X)
        return X

# lenet5 = LeNet5_R()
lenet5 = LeNet5_S()
if type(lenet5) == LeNet5_S:
    if isinstance(lenet5, LeNet5_S):
        print("\nlenet5 is an instance of LeNet5_S\n")
        for m in lenet5.modules(): # type: ignore
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif isinstance(lenet5, LeNet5_R):
        print("lenet5 is an instance of LeNet5_R")
    else:
        print(f"lenet5 is of type {type(lenet5)}")

lenet5.to(device=device)
print(lenet5)
    
print("\n\nLet's start Training:\n")
lenet5.train()

optimizer = torch.optim.SGD(lenet5.parameters(), lr=0.01, momentum=0.9)
loss = nn.CrossEntropyLoss()

epochs = 10
for epoch in range(epochs):
    epoch_loss = 0
    correct = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        pred = lenet5.forward(data)
        pred_loss = loss(pred, target)
        pred_loss.backward()
        epoch_loss += pred_loss
        optimizer.step()
        
        pred_correct = torch.argmax(pred, dim=1)
        correct += torch.sum(pred_correct == target)
        
    avg_loss = epoch_loss / len(train_loader)
    correct = correct / len(training_data.data)
    
    print(f"Epoch: {epoch+1} \t|\t Loss: {epoch_loss:.2f} \t|\t Avg Loss: {avg_loss:.2f} \t|\t Accuracy: {correct:.2f}")
    
    
print("\n\nLet's start evaluation:\n")
lenet5.eval()

epochs = 0
epoch_loss = 0
correct = 0
for batch_idx, (data, target) in enumerate(test_loader):
    data = data.to(device)
    target = target.to(device)
    
    pred = lenet5.forward(data)
    pred_loss = loss(pred, target)
    epoch_loss += pred_loss
    
    pred_correct = torch.argmax(pred, dim=1)
    correct += torch.sum(pred_correct == target)
    
avg_loss = epoch_loss / len(test_loader)
correct = correct / len(testing_data.data)

print(f"Epoch: {0} \t|\t Loss: {epoch_loss:.2f} \t|\t Avg Loss: {avg_loss:.2f} \t|\t Accuracy: {correct:.2f}")

end_time = time.time()
print(f"\n\nExceution completed in: {(end_time - start_time):.4f}")