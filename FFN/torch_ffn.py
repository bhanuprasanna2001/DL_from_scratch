# Now as I have implemented ffn from scratch just using numpy.
# It is time to build ffn using torch for the same mnist dataset with the
# same architecture with same hidden layers in pytorch.

import os
import torch
from torch import nn
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt

train_data = tv.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=tv.transforms.ToTensor()
)

test_data = tv.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=tv.transforms.ToTensor()
)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

class ffn(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.flatten = nn.Flatten()
        
        self.z1 = nn.Linear(784, 512)
        self.h1 = nn.ReLU()
        self.z2 = nn.Linear(512, 256)
        self.h2 = nn.ReLU()
        self.z3 = nn.Linear(256, 128)
        self.h3 = nn.ReLU()
        self.z4 = nn.Linear(128, 10)
        
        
    def forward(self, X):
        
        X = self.flatten(X)
        
        X = self.z1(X)
        X = self.h1(X)
        X = self.z2(X)
        X = self.h2(X)
        X = self.z3(X)
        X = self.h3(X)
        X = self.z4(X)
        
        return X
        

model = ffn().to(device=device)
print(model, "\n")

if __name__ == "__main__":
    epochs = 10
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    loss_history = []
    acc_history = []
    
    print("Training FFN on MNIST with PyTorch...\n")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        correct = 0
        for batch_id, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Make the data to device
            data = data.to(device)
            target = target.to(device)
            
            # Shape of the data
            if epoch == 0 and batch_id == 0:
                print(f"X Shape: {data.shape}, y Shape: {target.shape}\n")
            
            # Send the data through the forward process    
            pred = model.forward(data)
            
            # Calculate the loss
            pred_loss = loss_fn(pred, target)
            epoch_loss += pred_loss.item()
            
            # Backpropagate
            pred_loss.backward()
            optimizer.step()
            
            # Checking correctness
            pred_labels = torch.argmax(pred, dim=1)
            correct += (pred_labels == target).sum().item()
            
        avg_loss = epoch_loss / len(train_dataloader)
        accuracy = correct / len(train_dataloader.dataset) # type: ignore
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")
            
    print("\n" + "="*70)
    print("Training complete! Evaluating on test set...")
    print("="*70 + "\n")
                
    # Now we have trained the model, it is time to do testing
    model.eval()
    with torch.no_grad():
        test_loss = 0
        correct = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        for batch_id, (data, target) in enumerate(test_dataloader):
            data = data.to(device)
            target = target.to(device)
            
            if batch_id == 0:
                print(f"Test X Shape: {data.shape}, y Shape: {target.shape}\n")
                
            pred = model.forward(data)
            pred_probs = torch.softmax(pred, dim=1)
            pred_labels = torch.argmax(pred, dim=1)
            
            pred_loss = loss_fn(pred, target)
            test_loss += pred_loss.item()
            
            correct += (pred_labels == target).sum().item()
            
            all_preds.extend(pred_labels.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(pred_probs.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_dataloader)
        test_accuracy = correct / len(test_dataloader.dataset) # type: ignore
        print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    # Create output directory
    os.makedirs('output_figs/torch_ffn', exist_ok=True)
    
    # Plot training metrics
    print("Generating training metrics plots...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(loss_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    ax2.plot(acc_history)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('output_figs/torch_ffn/training_metrics.png', dpi=150)
    plt.close()
    print("Training metrics saved to output_figs/torch_ffn/training_metrics.png")
    
    # Confusion Matrix
    print("Generating confusion matrix...")
    confusion_mat = np.zeros((10, 10), dtype=int)
    for true, pred in zip(all_targets, all_preds):
        confusion_mat[true, pred] += 1
    
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_mat, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    for i in range(10):
        for j in range(10):
            plt.text(j, i, str(confusion_mat[i, j]), ha='center', va='center', 
                    color='white' if confusion_mat[i, j] > confusion_mat.max()/2 else 'black')
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.tight_layout()
    plt.savefig('output_figs/torch_ffn/confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to output_figs/torch_ffn/confusion_matrix.png")
    
    # Per-class metrics
    print("\nPer-class Metrics:")
    print("-" * 70)
    print(f"{'Class':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 70)
    
    for class_idx in range(10):
        tp = confusion_mat[class_idx, class_idx]
        fp = confusion_mat[:, class_idx].sum() - tp
        fn = confusion_mat[class_idx, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = confusion_mat[class_idx, :].sum()
        
        print(f"{class_idx:<8} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    print("-" * 70)
    
    # ROC and AUC curves
    print("\nGenerating ROC curves and calculating AUC...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    # One-hot encode true labels
    y_true_onehot = np.zeros((len(all_targets), 10))
    for i, label in enumerate(all_targets):
        y_true_onehot[i, label] = 1
    
    auc_scores = []
    for class_idx in range(10):
        # Calculate TPR and FPR
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            predictions = (all_probs[:, class_idx] >= threshold).astype(int)
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
        axes[class_idx].plot(fpr_list, tpr_list, label=f'AUC = {auc:.3f}')
        axes[class_idx].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        axes[class_idx].set_xlabel('False Positive Rate')
        axes[class_idx].set_ylabel('True Positive Rate')
        axes[class_idx].set_title(f'ROC Curve - Class {class_idx}')
        axes[class_idx].legend(loc='lower right')
        axes[class_idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output_figs/torch_ffn/roc_curves.png', dpi=150)
    plt.close()
    print("ROC curves saved to output_figs/torch_ffn/roc_curves.png")
    
    print("\nAUC Scores per class:")
    for class_idx, auc in enumerate(auc_scores):
        print(f"Class {class_idx}: {auc:.4f}")
    print(f"\nMean AUC: {np.mean(auc_scores):.4f}")
    
    print("\n" + "="*70)
    print("All metrics generated and saved to output_figs/torch_ffn/")
    print("="*70)
        
        