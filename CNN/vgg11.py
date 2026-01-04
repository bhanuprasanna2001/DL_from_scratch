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


# The VGG net is so similar and also too expensive to train, so I have not built it, because it is also the cnn layers stacked, that's it.
# This stacking introduces vanishing gradient problem which is resolved by the resnet, which I will be building.


end_time = time.time()
print(f"\n\nExceution completed in: {(end_time - start_time):.4f}")