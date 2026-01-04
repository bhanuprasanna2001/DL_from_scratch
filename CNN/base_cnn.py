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

class BASE_CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.l1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.l1_act = nn.ReLU()
        self.l1_pool = nn.MaxPool2d(2)
        
        self.l2 = nn.Conv2d(16, 8, kernel_size=2, stride=1, padding=1)
        self.l2_act = nn.ReLU()
        self.l2_pool = nn.MaxPool2d(2)
        
        self.flatten = nn.Flatten()
        
        self.d1 = nn.LazyLinear(128)
        self.d1_act = nn.ReLU()
        self.d2 = nn.Linear(128, 10)
    
    def forward(self, X):
        X = self.l1(X)
        X = self.l1_act(X)
        X = self.l1_pool(X)
        X = self.l2(X)
        X = self.l2_act(X)
        X = self.l2_pool(X)
        
        X = self.flatten(X)
        
        X = self.d1(X)
        X = self.d1_act(X)
        X = self.d2(X)
        
        return X
    
class BASE_CNN_BN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.l1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.l1_bn = nn.LazyBatchNorm2d()
        self.l1_act = nn.ReLU()
        
        self.l2 = nn.Conv2d(16, 8, kernel_size=2, stride=1, padding=1)
        self.l2_bn = nn.LazyBatchNorm2d()
        self.l2_act = nn.ReLU()
        
        self.flatten = nn.Flatten()
        
        self.d1 = nn.LazyLinear(128)
        self.d1_act = nn.ReLU()
        self.d2 = nn.Linear(128, 10)
    
    def forward(self, X):
        X = self.l1(X)
        X = self.l1_bn(X)
        X = self.l1_act(X)
        X = self.l2(X)
        X = self.l2_bn(X)
        X = self.l2_act(X)
        
        X = self.flatten(X)
        
        X = self.d1(X)
        X = self.d1_act(X)
        X = self.d2(X)
        
        return X

base_cnn = BASE_CNN_BN()
print(base_cnn)
base_cnn.to(device=device)

base_cnn.train()
optimizer = torch.optim.SGD(base_cnn.parameters())
loss = nn.CrossEntropyLoss()
        
print("\n\nLet's start training:\n")
epochs = 10
for i in range(epochs):
    epoch_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        pred = base_cnn.forward(data)
        pred_loss = loss(pred, target)
        pred_loss.backward()
        epoch_loss += pred_loss
        optimizer.step()
        
        pred_correct = torch.argmax(pred, dim=1)
        correct += torch.sum(pred_correct == target)
    
    correct = correct / len(training_data.data)
    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch: {i+1} \t|\t Loss: {epoch_loss:.2f} \t|\t Avg Loss: {avg_loss:.2f} \t|\t Accuracy: {correct:.2f}")
    
    
print("\n\nLet's start evaluation:\n")
base_cnn.eval()
eval_loss = 0
correct = 0
for batch_idx, (data, target) in enumerate(test_loader):
    data = data.to(device)
    target = target.to(device)
    
    pred = base_cnn.forward(data)
    pred_loss = loss(pred, target)
    
    eval_loss += pred_loss
    
    pred_correct = torch.argmax(pred, dim=1)
    correct += torch.sum(pred_correct == target)
    
correct = correct / len(testing_data.data)
avg_loss = eval_loss / len(test_loader)

print(f"Epoch: {0} \t|\t Loss: {eval_loss:.2f} \t|\t Avg Loss: {avg_loss:.2f} \t|\t Accuracy: {correct:.2f}")

end_time = time.time()
print(f"\n\nExceution completed in: {(end_time - start_time):.4f}")

