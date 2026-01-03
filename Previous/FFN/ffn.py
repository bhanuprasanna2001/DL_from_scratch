# I have previously implemented ML Algorithms from scratch to better understand the
# intiution behind each ML algorithm.
# To check that out: https://github.com/bhanuprasanna2001/ML_from_scratch

# Now it is time to get into Deep Learning.
# Starting with a feed forward neural network to predict MNIST dataset accuractely.

# The MNIST consists of images with handwritten numbers from 0 - 9 = 10 classes.
# Each image of 28x28 size, so 28x28 = 784.

# So, I will be building the following network:
#   1. Input layer - 784
#   2. Hidden Layer 1 - 512
#   3. Hidden Layer 2 - 256
#   4. Hidden Layer 3 - 128
#   5. Output Layer 4 - 10 (Softmax)

# Let's start implementation :)


# The MNIST Data loading was taken from: https://numpy.org/numpy-tutorials/tutorial-deep-learning-on-mnist/
import gzip
import os

import matplotlib.pyplot as plt
import numpy as np
import requests

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
    ),
    "\n",
)

# The FFN implementation:


def sparse_categorical_crossentropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # y_true is a scalar integer (class label)
    # y_pred is a vector of probabilities (10,)
    loss = -np.log(y_pred[y_true])

    return loss


def relu(x):
    return np.where(x > 0, x, 0)


def relu_backward(dout, x):
    dz = dout.copy()
    dz[x <= 0] = 0
    return dz


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


class FFN_MNIST:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

        # We know that the input layer is 784.
        # Using He initialization for ReLU networks: std = sqrt(2/n_in)

        # First hidden layer - Linear Layer (w1, b1)
        self.w1 = np.random.randn(784, 512) * np.sqrt(2.0 / 784)
        self.b1 = np.zeros(512)

        # Second hidden layer - Linear Layer (w2, b2)
        self.w2 = np.random.randn(512, 256) * np.sqrt(2.0 / 512)
        self.b2 = np.zeros(256)

        # Third hidden layer - Linear Layer (w3, b3)
        self.w3 = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
        self.b3 = np.zeros(128)

        # Output Layer
        self.w4 = np.random.randn(128, 10) * np.sqrt(2.0 / 128)
        self.b4 = np.zeros(10)

        # We will be using the relu activation function at each hidden layer
        # The output layer will be using the softmax layer

    def fit(self, X, y, epochs=10, subset_size=5000):
        # Use subset for faster training
        self.X = X[:subset_size].copy()
        self.y = y[:subset_size].copy()

        self.loss_history = []
        self.acc_history = []

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0

            for i in range(len(self.X)):
                y_pred = self._forward(self.X[i])
                epoch_loss += sparse_categorical_crossentropy(self.y[i], y_pred)
                self._backward(self.X[i], self.y[i], y_pred, epoch_loss)
                self._update_grads()

                if np.argmax(y_pred) == self.y[i]:
                    correct += 1

            avg_loss = epoch_loss / len(self.X)
            accuracy = correct / len(self.X)
            self.loss_history.append(avg_loss)
            self.acc_history.append(accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}"
            )

    def _forward(self, X):
        self.Z1 = (X @ self.w1) + self.b1
        self.H1 = relu(self.Z1)
        self.Z2 = (self.H1 @ self.w2) + self.b2
        self.H2 = relu(self.Z2)
        self.Z3 = (self.H2 @ self.w3) + self.b3
        self.H3 = relu(self.Z3)
        self.Z4 = (self.H3 @ self.w4) + self.b4
        out = softmax(self.Z4)

        return out

    def _backward(self, X, y_true, y_pred, loss):
        # First we obtained the loss, now we have to do reverse
        # Convert y_true to one-hot encoding
        y_true_onehot = np.zeros(10)
        y_true_onehot[y_true] = 1

        # Gradient of loss w.r.t. output (softmax derivative with cross-entropy)
        self.dz4 = y_pred - y_true_onehot  # Shape: (10,)

        # Gradient w.r.t. w4: outer product of H3 and dz4
        self.dw4 = np.outer(self.H3, self.dz4)  # Shape: (128, 10)
        self.db4 = self.dz4  # Shape: (10,)

        # Backprop to hidden layer 3
        dh3 = self.dz4 @ self.w4.T  # Shape: (128,)

        self.dz3 = relu_backward(dh3, self.Z3)  # Shape: (128,)
        self.dw3 = np.outer(self.H2, self.dz3)  # Shape: (256, 128)
        self.db3 = self.dz3  # Shape: (128,)

        # Backprop to hidden layer 2
        dh2 = self.dz3 @ self.w3.T  # Shape: (256,)

        self.dz2 = relu_backward(dh2, self.Z2)  # Shape: (256,)
        self.dw2 = np.outer(self.H1, self.dz2)  # Shape: (512, 256)
        self.db2 = self.dz2  # Shape: (256,)

        # Backprop to hidden layer 1
        dh1 = self.dz2 @ self.w2.T  # Shape: (512,)

        self.dz1 = relu_backward(dh1, self.Z1)  # Shape: (512,)
        self.dw1 = np.outer(X, self.dz1)  # Shape: (784, 512)
        self.db1 = self.dz1  # Shape: (512,)

    def _update_grads(self):
        self.w1 = self.w1 - (self.learning_rate * self.dw1)
        self.b1 = self.b1 - (self.learning_rate * self.db1)

        self.w2 = self.w2 - (self.learning_rate * self.dw2)
        self.b2 = self.b2 - (self.learning_rate * self.db2)

        self.w3 = self.w3 - (self.learning_rate * self.dw3)
        self.b3 = self.b3 - (self.learning_rate * self.db3)

        self.w4 = self.w4 - (self.learning_rate * self.dw4)
        self.b4 = self.b4 - (self.learning_rate * self.db4)

    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            y_pred = self._forward(X[i])
            predictions.append(np.argmax(y_pred))
        return np.array(predictions)

    def predict_proba(self, X):
        probabilities = []
        for i in range(len(X)):
            y_pred = self._forward(X[i])
            probabilities.append(y_pred)
        return np.array(probabilities)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy

    def plot_metrics(self):
        os.makedirs("output_figs/fnn", exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.loss_history)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True)

        ax2.plot(self.acc_history)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy")
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("output_figs/fnn/training_metrics.png", dpi=150)
        plt.close()
        print("Training metrics saved to output_figs/fnn/training_metrics.png")


if __name__ == "__main__":
    # Train the model
    print("Training FFN on MNIST...\n")
    ffn = FFN_MNIST(learning_rate=0.01)
    ffn.fit(x_train, y_train, epochs=10, subset_size=5000)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_subset = 1000
    y_pred = ffn.predict(x_test[:test_subset])
    y_true = y_test[:test_subset]
    y_proba = ffn.predict_proba(x_test[:test_subset])

    test_acc = np.mean(y_pred == y_true)
    print(f"Test Accuracy: {test_acc:.4f}\n")

    # Confusion Matrix
    print("Generating confusion matrix...")
    confusion_mat = np.zeros((10, 10), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion_mat[true, pred] += 1

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_mat, cmap="Blues", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            plt.text(
                j,
                i,
                str(confusion_mat[i, j]),
                ha="center",
                va="center",
                color="white"
                if confusion_mat[i, j] > confusion_mat.max() / 2
                else "black",
            )
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.savefig("output_figs/fnn/confusion_matrix.png", dpi=150)
    plt.close()
    print("Confusion matrix saved to output_figs/fnn/confusion_matrix.png")

    # Per-class metrics
    print("\nPer-class Metrics:")
    print("-" * 70)
    print(
        f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}"
    )
    print("-" * 70)

    for class_idx in range(10):
        tp = confusion_mat[class_idx, class_idx]
        fp = confusion_mat[:, class_idx].sum() - tp
        fn = confusion_mat[class_idx, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        support = confusion_mat[class_idx, :].sum()

        print(
            f"{class_idx:<8} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}"
        )

    print("-" * 70)

    # ROC and AUC curves
    print("\nGenerating ROC curves and calculating AUC...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    # One-hot encode true labels
    y_true_onehot = np.zeros((len(y_true), 10))
    for i, label in enumerate(y_true):
        y_true_onehot[i, label] = 1

    auc_scores = []
    for class_idx in range(10):
        # Calculate TPR and FPR
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []

        for threshold in thresholds:
            predictions = (y_proba[:, class_idx] >= threshold).astype(int)
            tp = np.sum((predictions == 1) & (y_true_onehot[:, class_idx] == 1))
            fp = np.sum((predictions == 1) & (y_true_onehot[:, class_idx] == 0))
            tn = np.sum((predictions == 0) & (y_true_onehot[:, class_idx] == 0))
            fn = np.sum((predictions == 0) & (y_true_onehot[:, class_idx] == 1))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            tpr_list.append(tpr)
            fpr_list.append(fpr)

        # Calculate AUC using trapezoidal rule
        fpr_array = np.array(fpr_list)
        tpr_array = np.array(tpr_list)
        sorted_indices = np.argsort(fpr_array)
        fpr_sorted = fpr_array[sorted_indices]
        tpr_sorted = tpr_array[sorted_indices]
        auc = np.trapz(tpr_sorted, fpr_sorted)
        auc_scores.append(auc)

        # Plot ROC curve
        axes[class_idx].plot(fpr_list, tpr_list, label=f"AUC = {auc:.3f}")
        axes[class_idx].plot([0, 1], [0, 1], "k--", alpha=0.3)
        axes[class_idx].set_xlabel("False Positive Rate")
        axes[class_idx].set_ylabel("True Positive Rate")
        axes[class_idx].set_title(f"ROC Curve - Class {class_idx}")
        axes[class_idx].legend(loc="lower right")
        axes[class_idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output_figs/fnn/roc_curves.png", dpi=150)
    plt.close()
    print("ROC curves saved to output_figs/fnn/roc_curves.png")

    print("\nAUC Scores per class:")
    for class_idx, auc in enumerate(auc_scores):
        print(f"Class {class_idx}: {auc:.4f}")
    print(f"\nMean AUC: {np.mean(auc_scores):.4f}")

    # Plot training metrics
    print("\nSaving training metrics...")
    ffn.plot_metrics()

    print("\n" + "=" * 70)
    print("All metrics generated and saved to output_figs/fnn/")
    print("=" * 70)
