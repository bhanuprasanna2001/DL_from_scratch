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




end_time = time.time()
print(f"\n\nExceution completed in: {(end_time - start_time):.4f}")