# Deep Learning from Scratch

A comprehensive implementation of fundamental deep learning architectures using NumPy and PyTorch, focusing on mathematical foundations and practical applications.

## Overview

This repository contains from-scratch implementations of core deep learning models, progressing from simple feed-forward networks to advanced architectures like variational autoencoders and sequence-to-sequence models. Each implementation emphasizes understanding the underlying mathematics and gradient computations.

For in blog format refer to: [BLOG DL](https://bhanuprasanna2001.github.io/learning/ai/DL/)

## Implemented Architectures

### Feed-Forward Networks (FFN)
- **NumPy Implementation**: Gradient computation with backpropagation
- **PyTorch Implementation**
- **Dataset**: MNIST digit classification
- **Features**: Cross-entropy loss, ReLU activation, softmax output, ROC curves, multi-class metrics

### Convolutional Neural Networks (CNN)
- **Custom CNN**: From-scratch convolution operations with NumPy
- **Modern Architectures**: LeNet-5, AlexNet, VGG-11, ResNet
- **Datasets**: MNIST, CIFAR-100
- **Features**: Batch normalization, pooling layers, data augmentation, various optimizers

### Recurrent Neural Networks (RNN)
- **NumPy Implementation**: Manual BPTT (Backpropagation Through Time)
- **Task**: Sequence prediction (next number in sequence)
- **Features**: Hidden state management, temporal gradient flow

### Long Short-Term Memory (LSTM)
- **NumPy Implementation**: Full LSTM cell mechanics (gates: input, forget, output)
- **Applications**:
  - **Baby Name Generation**: Character-level language model with sampling strategies (temperature, top-k, top-p)
  - **Seq2Seq Translation**: English-to-French translation with attention mechanism
- **Features**: Packed sequences, teacher forcing, BLEU score evaluation

### Autoencoders
Three distinct implementations exploring dimensionality reduction:
- **CNN Autoencoder**: MNIST reconstruction with bottleneck visualization
- **LSTM Autoencoder**: Time-series reconstruction (ECG data)
- **Denoising Autoencoder**: Fashion-MNIST noise removal

### Variational Autoencoders (VAE)
- **β-VAE Implementation**: Controllable latent disentanglement
- **Dataset**: MNIST
- **Features**: KL divergence regularization, latent space interpolation, reconstruction quality analysis
- **Mathematics**: Evidence Lower Bound (ELBO) optimization with reparameterization trick

## Project Structure

```
.
├── FFN/                    # Feed-forward networks
│   ├── ffn.py             # NumPy implementation
│   └── torch_ffn.py       # PyTorch implementation
├── CNN/                    # Convolutional networks
│   ├── cnn.py             # Custom CNN
│   ├── base_cnn.py        # Basic architecture
│   ├── lenet5.py          # LeNet-5
│   ├── alexnet.py         # AlexNet
│   ├── vgg11.py           # VGG-11
│   └── resnet.py          # ResNet
├── RNN/                    # Recurrent networks
│   └── rnn.py             # NumPy implementation
├── LSTM/                   # Long short-term memory
│   ├── lstm.py            # NumPy LSTM
│   ├── name_gen.py        # Name generation
│   └── seq2seq.py         # Translation model
├── Autoencoders/          # Autoencoder variants
│   ├── cnn_autoencoder.py
│   ├── lstm_autoencoder.py
│   └── noise_autoencoder.py
└── VAE/                    # Variational autoencoders
    ├── vae.py
    └── visualize_vae.py
```

## Key Features

- **From-Scratch Implementations**: Core algorithms implemented in NumPy to understand backpropagation mechanics
- **Mathematical Rigor**: Detailed comments explaining gradient computations and architectural choices
- **Comprehensive Evaluation**: Multiple metrics, visualization tools, and performance analysis
- **Progressive Complexity**: Structured learning path from basic FFN to advanced VAE architectures

## Requirements

```bash
numpy
torch
torchvision
matplotlib
pandas
sacrebleu  # For translation evaluation
spacy      # For text tokenization
```

## Usage

Each module is self-contained and can be run independently:

```bash
# Train feed-forward network
python FFN/ffn.py

# Train CNN on MNIST
python CNN/base_cnn.py

# Generate baby names
python LSTM/name_gen.py

# Train variational autoencoder
python VAE/vae.py
```

## References

- Optimization techniques: [Ruder (2016) - Optimizing Gradient Descent](https://www.ruder.io/optimizing-gradient-descent/)

## License

Open.
