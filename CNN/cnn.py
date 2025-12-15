# Here I will be implementing CNN from scratch.

# This will obviously be sphagetti code.

# The MNIST Data loading was taken from: https://numpy.org/numpy-tutorials/tutorial-deep-learning-on-mnist/

import os
import gzip
import requests
import numpy as np
import scipy
import matplotlib.pyplot as plt

np.random.seed(42)

data_dir = "../Data/MNIST"
os.makedirs(data_dir, exist_ok=True)

base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"

data_sources = {
    "training_images": "train-images-idx3-ubyte.gz",  # 60,000 training images.
    "test_images": "t10k-images-idx3-ubyte.gz",  # 10,000 test images.
    "training_labels": "train-labels-idx1-ubyte.gz",  # 60,000 training labels.
    "test_labels": "t10k-labels-idx1-ubyte.gz",  # 10,000 test labels.
}

for fname in data_sources.values():
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        print("Downloading file: " + fname)
        resp = requests.get(base_url + fname, stream=True)
        resp.raise_for_status()  # Ensure download was succesful
        with open(fpath, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=128):
                fh.write(chunk)

mnist_dataset = {}

# Images
for key in ("training_images", "test_images"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(
            mnist_file.read(), np.uint8, offset=16
        ).reshape(-1, 28 * 28)
# Labels
for key in ("training_labels", "test_labels"):
    with gzip.open(os.path.join(data_dir, data_sources[key]), "rb") as mnist_file:
        mnist_dataset[key] = np.frombuffer(mnist_file.read(), np.uint8, offset=8)
        
x_train, y_train, x_test, y_test = (
    mnist_dataset["training_images"],
    mnist_dataset["training_labels"],
    mnist_dataset["test_images"],
    mnist_dataset["test_labels"],
)

# Normalize pixel values to [0, 1] range
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print(
    "The shape of training images: {} and training labels: {}".format(
        x_train.shape, y_train.shape
    )
)
print(
    "The shape of test images: {} and test labels: {}".format(
        x_test.shape, y_test.shape
    ), "\n"
)

def print_mnist(X, y):
    C, H, W = X.shape
    k = 0  # show first channel
    for i in range(H):
        for j in range(W):
            print(f"{X[k,i,j]:.4f}", end="\t")
        print()
    print(f"\n{y}")

# Let's start implementing Convolution Layer from scratch.

class CNN2D:
    
    def __init__(self, in_channels=1, out_channels=32, kernel=2):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        
        self.weights = np.random.randn(self.out_channels, self.in_channels, self.kernel, self.kernel)
        self.bias = np.random.randn(self.out_channels)
        
        
    def forward(self, X):
        self.input_data = X.copy()
        
        print(self.weights.shape, self.bias.shape, self.input_data.shape)
        
        out = []
        
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                out.append(scipy.signal.correlate2d(self.input_data[j], self.weights[i,j], 'valid'))
        
        out = np.array(out)
        print(self.bias[:, None, None].shape, out.shape)
        out += self.bias[:, None, None]
        
        return out
        
    
    def backward(self, out_gradient, lr=0.001):
        kernel_gradient = np.zeros_like(self.weights)
        input_gradient = np.zeros_like(self.input_data)
        
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                kernel_gradient[i,j] = scipy.signal.correlate2d(self.input_data[j], out_gradient[i], "valid")
                input_gradient[j] += scipy.signal.convolve2d(out_gradient[i], self.weights[i,j], 'full')
                
        self.weights -= lr * kernel_gradient
        self.bias -= lr * np.sum(out_gradient, axis=(1,2))
        
        return input_gradient
    
# The output size is formulated like this:
# output_size = ((input + 2 * padding - kernel) / stride) + 1
        
conv_1 = CNN2D()
        
for data, target in zip(x_train, y_train):
    data = np.reshape(data, (1,28,28))
    conv_1.forward(data)
    
    break
    