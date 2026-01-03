# Here I will be implementing CNN from scratch.

# This will obviously be sphagetti code.

# The MNIST Data loading was taken from: https://numpy.org/numpy-tutorials/tutorial-deep-learning-on-mnist/

import os
import gzip
import requests
import numpy as np
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
    
    def __init__(self, in_channels=1, out_channels=32, kernel=3, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        
        # Xavier initialization
        self.weights = np.random.randn(self.out_channels, self.in_channels, self.kernel, self.kernel) * np.sqrt(2.0 / (in_channels * kernel * kernel))
        self.bias = np.zeros(self.out_channels)
        
    def _pad_input(self, X):
        """Apply zero padding to input."""
        if self.padding == 0:
            return X
        return np.pad(X, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
    
    def _correlate2d(self, input_slice, kernel):
        """Numpy-only implementation of 2D correlation."""
        H, W = input_slice.shape
        kH, kW = kernel.shape
        out_H = (H - kH) // self.stride + 1
        out_W = (W - kW) // self.stride + 1
        
        output = np.zeros((out_H, out_W))
        
        for i in range(out_H):
            for j in range(out_W):
                h_start = i * self.stride
                w_start = j * self.stride
                receptive_field = input_slice[h_start:h_start+kH, w_start:w_start+kW]
                output[i, j] = np.sum(receptive_field * kernel)
        
        return output
    
    def _convolve2d_full(self, gradient, kernel):
        """Numpy-only implementation of 2D convolution (full mode)."""
        gH, gW = gradient.shape
        kH, kW = kernel.shape
        
        # For full convolution, pad the gradient
        pad_h = kH - 1
        pad_w = kW - 1
        padded_grad = np.pad(gradient, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        out_H = gH + kH - 1
        out_W = gW + kW - 1
        output = np.zeros((out_H, out_W))
        
        # Flip kernel for convolution
        flipped_kernel = np.flip(kernel)
        
        for i in range(out_H):
            for j in range(out_W):
                receptive_field = padded_grad[i:i+kH, j:j+kW]
                output[i, j] = np.sum(receptive_field * flipped_kernel)
        
        return output
        
    def forward(self, X):
        """Forward pass with stride and padding support."""
        self.input_data = X.copy()
        self.padded_input = self._pad_input(X)
        
        C_in, H_padded, W_padded = self.padded_input.shape
        out_H = (H_padded - self.kernel) // self.stride + 1
        out_W = (W_padded - self.kernel) // self.stride + 1
        
        output = np.zeros((self.out_channels, out_H, out_W))
        
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                output[i] += self._correlate2d(self.padded_input[j], self.weights[i, j])
            output[i] += self.bias[i]
        
        return output
    
    def backward(self, out_gradient, lr=0.001):
        """Backward pass with stride and padding support."""
        kernel_gradient = np.zeros_like(self.weights)
        input_gradient = np.zeros_like(self.padded_input)
        
        # If stride > 1, we need to upsample the gradient
        if self.stride > 1:
            upsampled_grad = np.zeros((self.out_channels, 
                                       (out_gradient.shape[1]-1)*self.stride + 1,
                                       (out_gradient.shape[2]-1)*self.stride + 1))
            for i in range(out_gradient.shape[1]):
                for j in range(out_gradient.shape[2]):
                    upsampled_grad[:, i*self.stride, j*self.stride] = out_gradient[:, i, j]
            out_gradient = upsampled_grad
        
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                # Gradient w.r.t. weights
                kernel_gradient[i, j] = self._correlate2d(self.padded_input[j], out_gradient[i])
                # Gradient w.r.t. input
                input_gradient[j] += self._convolve2d_full(out_gradient[i], self.weights[i, j])
        
        # Update weights and bias
        self.weights -= lr * kernel_gradient
        self.bias -= lr * np.sum(out_gradient, axis=(1, 2))
        
        # Remove padding from input gradient
        if self.padding > 0:
            input_gradient = input_gradient[:, self.padding:-self.padding, self.padding:-self.padding]
        
        return input_gradient
    
# The output size is formulated like this:
# output_size = ((input + 2 * padding - kernel) / stride) + 1

class ReLU:
    def forward(self, X):
        self.input_data = X
        return np.maximum(0, X)
    
    def backward(self, out_gradient):
        return out_gradient * (self.input_data > 0)

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, X):
        self.input_data = X
        C, H, W = X.shape
        out_H = (H - self.pool_size) // self.stride + 1
        out_W = (W - self.pool_size) // self.stride + 1
        
        output = np.zeros((C, out_H, out_W))
        self.max_indices = np.zeros((C, out_H, out_W, 2), dtype=int)
        
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * self.stride
                    w_start = j * self.stride
                    receptive_field = X[c, h_start:h_start+self.pool_size, w_start:w_start+self.pool_size]
                    max_val = np.max(receptive_field)
                    output[c, i, j] = max_val
                    
                    # Store max index for backward pass
                    max_idx = np.unravel_index(np.argmax(receptive_field), receptive_field.shape)
                    self.max_indices[c, i, j] = [h_start + max_idx[0], w_start + max_idx[1]]
        
        return output
    
    def backward(self, out_gradient):
        input_gradient = np.zeros_like(self.input_data)
        C, out_H, out_W = out_gradient.shape
        
        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    h_idx, w_idx = self.max_indices[c, i, j]
                    input_gradient[c, h_idx, w_idx] += out_gradient[c, i, j]
        
        return input_gradient

class Flatten:
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(1, -1)
    
    def backward(self, out_gradient):
        return out_gradient.reshape(self.input_shape)

class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size)
    
    def forward(self, X):
        self.input_data = X
        return np.dot(X, self.weights) + self.bias
    
    def backward(self, out_gradient, lr=0.001):
        weights_gradient = np.dot(self.input_data.T, out_gradient)
        input_gradient = np.dot(out_gradient, self.weights.T)
        
        self.weights -= lr * weights_gradient
        self.bias -= lr * np.sum(out_gradient, axis=0)
        
        return input_gradient

class Softmax:
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        self.output = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.output
    
    def backward(self, y_true):
        # Combined softmax + cross-entropy gradient
        return self.output - y_true

def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_pred.shape[0]

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

# Build a simple CNN
print("Building CNN model...")
conv1 = CNN2D(in_channels=1, out_channels=8, kernel=3, stride=1, padding=1)
relu1 = ReLU()
pool1 = MaxPool2D(pool_size=2, stride=2)

conv2 = CNN2D(in_channels=8, out_channels=16, kernel=3, stride=1, padding=1)
relu2 = ReLU()
pool2 = MaxPool2D(pool_size=2, stride=2)

flatten = Flatten()
dense1 = Dense(16 * 7 * 7, 128)
relu3 = ReLU()
dense2 = Dense(128, 10)
softmax = Softmax()

print("\nStarting training...\n")

# Training parameters
num_epochs = 3
batch_size = 32
learning_rate = 0.01
num_samples = 1000  # Use subset for faster training

# Use subset of data
x_train_subset = x_train[:num_samples]
y_train_subset = y_train[:num_samples]

for epoch in range(num_epochs):
    epoch_loss = 0
    correct = 0
    
    # Shuffle data
    indices = np.random.permutation(num_samples)
    x_train_subset = x_train_subset[indices]
    y_train_subset = y_train_subset[indices]
    
    for i in range(0, num_samples, batch_size):
        batch_x = x_train_subset[i:i+batch_size]
        batch_y = y_train_subset[i:i+batch_size]
        
        batch_loss = 0
        
        for data, target in zip(batch_x, batch_y):
            # Reshape to (C, H, W)
            data = data.reshape(1, 28, 28)
            
            # Forward pass
            out = conv1.forward(data)
            out = relu1.forward(out)
            out = pool1.forward(out)
            
            out = conv2.forward(out)
            out = relu2.forward(out)
            out = pool2.forward(out)
            
            out = flatten.forward(out)
            out = dense1.forward(out)
            out = relu3.forward(out)
            out = dense2.forward(out)
            out = softmax.forward(out)
            
            # Calculate loss
            y_true = one_hot_encode(np.array([target]))
            loss = cross_entropy_loss(out, y_true)
            batch_loss += loss
            
            # Check accuracy
            if np.argmax(out) == target:
                correct += 1
            
            # Backward pass
            grad = softmax.backward(y_true)
            grad = dense2.backward(grad, learning_rate)
            grad = relu3.backward(grad)
            grad = dense1.backward(grad, learning_rate)
            grad = flatten.backward(grad)
            
            grad = pool2.backward(grad)
            grad = relu2.backward(grad)
            grad = conv2.backward(grad, learning_rate)
            
            grad = pool1.backward(grad)
            grad = relu1.backward(grad)
            grad = conv1.backward(grad, learning_rate)
        
        epoch_loss += batch_loss
        
        if (i // batch_size) % 5 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size + 1}/{num_samples//batch_size}, Loss: {batch_loss/batch_size:.4f}")
    
    accuracy = correct / num_samples * 100
    print(f"\nEpoch {epoch+1} - Avg Loss: {epoch_loss/num_samples:.4f}, Accuracy: {accuracy:.2f}%\n")

print("\nTraining complete!")

# Test on a few samples
print("\nTesting on 10 samples:")
test_correct = 0
for i in range(10):
    data = x_test[i].reshape(1, 28, 28)
    target = y_test[i]
    
    # Forward pass
    out = conv1.forward(data)
    out = relu1.forward(out)
    out = pool1.forward(out)
    out = conv2.forward(out)
    out = relu2.forward(out)
    out = pool2.forward(out)
    out = flatten.forward(out)
    out = dense1.forward(out)
    out = relu3.forward(out)
    out = dense2.forward(out)
    out = softmax.forward(out)
    
    pred = np.argmax(out)
    print(f"Sample {i+1}: True={target}, Predicted={pred}, {'✓' if pred == target else '✗'}")
    if pred == target:
        test_correct += 1

print(f"\nTest Accuracy (10 samples): {test_correct/10*100:.2f}%")
    