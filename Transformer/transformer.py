"""
Transformer Implementation from Scratch
========================================

Implementation of the Transformer architecture as described
in "Attention is All You Need" (Vaswani et al., 2017).

Task: We'll train the model to reverse sequences (e.g., [1,2,3,4] -> [4,3,2,1])
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# 1. POSITIONAL ENCODING
# ============================================================================
class PositionalEncoding(nn.Module):
    """
    Positional Encoding adds information about the position of tokens in a sequence.
    
    Since Transformers don't have recurrence or convolution, they need explicit
    position information. This uses sine and cosine functions of different frequencies.
    
    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    where:
        - pos is the position in the sequence
        - i is the dimension index
        - d_model is the embedding dimension
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        Args:
            d_model: Dimension of the embeddings/model
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create a matrix of shape (max_len, d_model) to hold positional encodings
        pe = torch.zeros(max_len, d_model)
        
        # Create a column vector of positions [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Create the division term for the formula
        # This creates [10000^(0/d_model), 10000^(2/d_model), ..., 10000^(d_model/d_model)]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices in the array (2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices in the array (2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: shape becomes (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (not a parameter, but part of the model state)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added, same shape as input
        """
        # Add positional encoding to the input
        # x.size(1) is the sequence length
        x = x + self.pe[:, :x.size(1), :]
        return x


# ============================================================================
# 2. MULTI-HEAD ATTENTION
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    The attention mechanism allows the model to focus on different parts of the input.
    Multi-head attention runs multiple attention operations in parallel and concatenates
    their outputs.
    
    The core idea:
        1. For each head, project Q, K, V using learned linear transformations
        2. Compute attention scores: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        3. Concatenate all heads and project back
    
    Intuition: Different heads can learn to attend to different types of relationships
    (e.g., one head for syntax, another for semantics).
    """
    
    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: Dimension of the model (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        # These transform the input into queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model)  # Query projection
        self.W_k = nn.Linear(d_model, d_model)  # Key projection
        self.W_v = nn.Linear(d_model, d_model)  # Value projection
        
        # Final linear layer after concatenating heads
        self.W_o = nn.Linear(d_model, d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
        
        Args:
            Q: Queries of shape (batch_size, num_heads, seq_len, d_k)
            K: Keys of shape (batch_size, num_heads, seq_len, d_k)
            V: Values of shape (batch_size, num_heads, seq_len, d_k)
            mask: Optional mask of shape (batch_size, 1, seq_len, seq_len)
        
        Returns:
            Attention output and attention weights
        """
        # Compute attention scores by matrix multiplication of Q and K^T
        # Shape: (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided (used for padding or future positions)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        # This normalizes scores to probabilities
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Multiply attention weights by V to get the output
        # Shape: (batch_size, num_heads, seq_len, d_k)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def split_heads(self, x):
        """
        Split the last dimension into (num_heads, d_k).
        Reshape from (batch_size, seq_len, d_model) 
                  to (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
        Combine heads back together.
        Reshape from (batch_size, num_heads, seq_len, d_k)
                  to (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of multi-head attention.
        
        Args:
            Q: Queries of shape (batch_size, seq_len, d_model)
            K: Keys of shape (batch_size, seq_len, d_model)
            V: Values of shape (batch_size, seq_len, d_model)
            mask: Optional mask
        
        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        # Apply linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply scaled dot-product attention
        attn_output, attn_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads
        output = self.combine_heads(attn_output)
        
        # Apply final linear projection
        output = self.W_o(output)
        
        return output


# ============================================================================
# 3. FEED-FORWARD NETWORK
# ============================================================================
class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    This is a simple two-layer neural network applied to each position independently.
    
    Formula: FFN(x) = max(0, xW1 + b1)W2 + b2
    
    The network expands the dimension (typically by 4x) and then contracts it back.
    This adds non-linearity and allows the model to learn complex transformations.
    """
    
    def __init__(self, d_model, d_ff):
        """
        Args:
            d_model: Dimension of the model
            d_ff: Dimension of the feed-forward layer (typically 4 * d_model)
        """
        super(FeedForward, self).__init__()
        
        # First linear layer: expand dimension
        self.linear1 = nn.Linear(d_model, d_ff)
        
        # Second linear layer: contract back to d_model
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # ReLU activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        """
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        # Apply first linear layer and ReLU
        x = self.relu(self.linear1(x))
        
        # Apply second linear layer
        x = self.linear2(x)
        
        return x


# ============================================================================
# 4. ENCODER LAYER
# ============================================================================
class EncoderLayer(nn.Module):
    """
    A single Encoder layer consists of:
        1. Multi-Head Self-Attention
        2. Add & Norm (residual connection + layer normalization)
        3. Feed-Forward Network
        4. Add & Norm (residual connection + layer normalization)
    
    Residual connections help with gradient flow during training.
    Layer normalization stabilizes the learning process.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout probability
        """
        super(EncoderLayer, self).__init__()
        
        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # Layer normalization (applied after each sub-layer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        
        Returns:
            Output of shape (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection and normalization
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Add & Norm
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))  # Add & Norm
        
        return x


# ============================================================================
# 5. DECODER LAYER
# ============================================================================
class DecoderLayer(nn.Module):
    """
    A single Decoder layer consists of:
        1. Masked Multi-Head Self-Attention (to prevent looking at future tokens)
        2. Add & Norm
        3. Multi-Head Cross-Attention (attending to encoder output)
        4. Add & Norm
        5. Feed-Forward Network
        6. Add & Norm
    
    The masked self-attention ensures that predictions for position i can only
    depend on the known outputs at positions less than i.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Dimension of the model
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout probability
        """
        super(DecoderLayer, self).__init__()
        
        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        
        # Cross-attention (attending to encoder output)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        
        # Feed-forward network
        self.feed_forward = FeedForward(d_model, d_ff)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Decoder input of shape (batch_size, tgt_seq_len, d_model)
            enc_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Mask for encoder output (padding mask)
            tgt_mask: Mask for decoder input (padding + look-ahead mask)
        
        Returns:
            Output of shape (batch_size, tgt_seq_len, d_model)
        """
        # Masked self-attention with residual connection and normalization
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with residual connection and normalization
        # Q comes from decoder, K and V come from encoder
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward with residual connection and normalization
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


# ============================================================================
# 6. ENCODER
# ============================================================================
class Encoder(nn.Module):
    """
    The Encoder consists of:
        1. Input Embedding
        2. Positional Encoding
        3. Stack of N Encoder Layers
    
    The encoder processes the source sequence and creates a representation
    that the decoder can attend to.
    """
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            num_layers: Number of encoder layers
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout probability
        """
        super(Encoder, self).__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for embeddings (as per the paper)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            mask: Optional attention mask
        
        Returns:
            Encoder output of shape (batch_size, seq_len, d_model)
        """
        # Embed tokens and scale by sqrt(d_model)
        x = self.embedding(x) * self.scale
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


# ============================================================================
# 7. DECODER
# ============================================================================
class Decoder(nn.Module):
    """
    The Decoder consists of:
        1. Output Embedding
        2. Positional Encoding
        3. Stack of N Decoder Layers
        4. Final Linear layer (to project to vocabulary size)
    
    The decoder generates the output sequence one token at a time,
    attending to the encoder output at each step.
    """
    
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, dropout=0.1):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model
            num_layers: Number of decoder layers
            num_heads: Number of attention heads
            d_ff: Dimension of feed-forward layer
            dropout: Dropout probability
        """
        super(Decoder, self).__init__()
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final linear layer to project to vocabulary
        self.fc_out = nn.Linear(d_model, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for embeddings
        self.scale = math.sqrt(d_model)
    
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target token indices of shape (batch_size, tgt_seq_len)
            enc_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Mask for encoder output
            tgt_mask: Mask for decoder input
        
        Returns:
            Logits of shape (batch_size, tgt_seq_len, vocab_size)
        """
        # Embed tokens and scale
        x = self.embedding(x) * self.scale
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        # Project to vocabulary size
        output = self.fc_out(x)
        
        return output


# ============================================================================
# 8. TRANSFORMER
# ============================================================================
class Transformer(nn.Module):
    """
    Complete Transformer model combining Encoder and Decoder.
    
    This is the main model class that brings everything together.
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, dropout=0.1):
        """
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Dimension of the model (default: 512)
            num_layers: Number of encoder/decoder layers (default: 6)
            num_heads: Number of attention heads (default: 8)
            d_ff: Dimension of feed-forward layer (default: 2048)
            dropout: Dropout probability (default: 0.1)
        """
        super(Transformer, self).__init__()
        
        # Encoder
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
        
        # Decoder
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, dropout)
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generate a mask to prevent attending to future positions.
        
        This is crucial for the decoder to ensure that predictions for position i
        can only depend on known outputs at positions less than i.
        
        Args:
            sz: Size of the mask (sequence length)
        
        Returns:
            Lower triangular mask of shape (sz, sz)
        """
        mask = torch.tril(torch.ones(sz, sz))
        return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    
    def forward(self, src, tgt):
        """
        Forward pass of the Transformer.
        
        Args:
            src: Source sequence of shape (batch_size, src_seq_len)
            tgt: Target sequence of shape (batch_size, tgt_seq_len)
        
        Returns:
            Output logits of shape (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # Generate masks
        # Target mask prevents looking at future tokens
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Encode source sequence
        enc_output = self.encoder(src)
        
        # Decode target sequence
        output = self.decoder(tgt, enc_output, tgt_mask=tgt_mask)
        
        return output


# ============================================================================
# 9. DATASET
# ============================================================================
class ReverseDataset(Dataset):
    """
    Simple dataset for sequence reversal task.
    
    Given a sequence like [1, 2, 3, 4], the model should learn to output [4, 3, 2, 1].
    
    This is a simple task that's easy to understand and verify, making it perfect
    for learning how Transformers work.
    """
    
    def __init__(self, num_samples=10000, seq_len=10, vocab_size=20):
        """
        Args:
            num_samples: Number of sequences to generate
            seq_len: Length of each sequence
            vocab_size: Size of vocabulary (number of unique tokens)
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Special tokens
        self.PAD_IDX = 0  # Padding token
        self.SOS_IDX = vocab_size - 2  # Start of sequence token
        self.EOS_IDX = vocab_size - 1  # End of sequence token
        
        # Generate dataset
        self.data = []
        for _ in range(num_samples):
            # Generate random sequence (excluding special tokens)
            seq = np.random.randint(1, vocab_size - 2, size=seq_len)
            
            # Source: add SOS and EOS tokens
            src = np.concatenate([[self.SOS_IDX], seq, [self.EOS_IDX]])
            
            # Target: reversed sequence with SOS and EOS
            tgt_input = np.concatenate([[self.SOS_IDX], seq[::-1]])
            tgt_output = np.concatenate([seq[::-1], [self.EOS_IDX]])
            
            self.data.append((src, tgt_input, tgt_output))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns:
            src: Source sequence (with SOS and EOS)
            tgt_input: Target input for decoder (with SOS)
            tgt_output: Target output for loss computation (with EOS)
        """
        src, tgt_input, tgt_output = self.data[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt_input, dtype=torch.long),
            torch.tensor(tgt_output, dtype=torch.long)
        )


# ============================================================================
# 10. TRAINING
# ============================================================================
def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    Train for one epoch.
    
    Args:
        model: Transformer model
        dataloader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on (CPU or GPU)
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (src, tgt_input, tgt_output) in enumerate(dataloader):
        # Move data to device
        src = src.to(device)
        tgt_input = tgt_input.to(device)
        tgt_output = tgt_output.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt_input)
        
        # Reshape output and target for loss computation
        # output shape: (batch_size, seq_len, vocab_size)
        # tgt_output shape: (batch_size, seq_len)
        output = output.view(-1, output.size(-1))
        tgt_output = tgt_output.view(-1)
        
        # Compute loss
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        total_loss += loss.item()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: Transformer model
        dataloader: DataLoader for validation data
        criterion: Loss function
        device: Device to evaluate on
    
    Returns:
        Average loss and accuracy
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for src, tgt_input, tgt_output in dataloader:
            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_output = tgt_output.to(device)
            
            # Forward pass
            output = model(src, tgt_input)
            
            # Compute loss
            output_flat = output.view(-1, output.size(-1))
            tgt_output_flat = tgt_output.view(-1)
            loss = criterion(output_flat, tgt_output_flat)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = output.argmax(dim=-1)
            correct += (predictions == tgt_output).sum().item()
            total += tgt_output.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def test_model(model, dataset, device, num_examples=5):
    """
    Test the model with a few examples and print results.
    
    Args:
        model: Trained Transformer model
        dataset: Dataset to sample from
        device: Device to run on
        num_examples: Number of examples to test
    """
    model.eval()
    print("\n" + "="*60)
    print("TESTING THE MODEL")
    print("="*60)
    
    with torch.no_grad():
        for i in range(num_examples):
            src, tgt_input, tgt_output = dataset[i]
            src = src.unsqueeze(0).to(device)
            tgt_input = tgt_input.unsqueeze(0).to(device)
            
            # Get model prediction
            output = model(src, tgt_input)
            predictions = output.argmax(dim=-1).squeeze(0)
            
            # Convert to lists for printing (excluding special tokens)
            src_tokens = src.squeeze(0).cpu().numpy()[1:-1]  # Remove SOS and EOS
            pred_tokens = predictions.cpu().numpy()[:-1]  # Remove EOS prediction
            target_tokens = tgt_output.numpy()[:-1]  # Remove EOS
            
            print(f"\nExample {i+1}:")
            print(f"  Input:      {src_tokens.tolist()}")
            print(f"  Target:     {target_tokens.tolist()}")
            print(f"  Predicted:  {pred_tokens.tolist()}")
            print(f"  Correct: {'✓' if np.array_equal(pred_tokens, target_tokens) else '✗'}")


# ============================================================================
# 11. MAIN FUNCTION
# ============================================================================
def main():
    """
    Main training script.
    """
    print("="*60)
    print("TRANSFORMER FROM SCRATCH - SEQUENCE REVERSAL TASK")
    print("="*60)
    
    # ========================================
    # Hyperparameters
    # ========================================
    # Dataset parameters
    NUM_SAMPLES = 5000       # Number of training samples
    SEQ_LEN = 8              # Length of sequences
    VOCAB_SIZE = 20          # Size of vocabulary
    
    # Model parameters (kept small for easy learning and fast training)
    D_MODEL = 128            # Embedding dimension
    NUM_LAYERS = 2           # Number of encoder/decoder layers
    NUM_HEADS = 4            # Number of attention heads
    D_FF = 512               # Feed-forward dimension
    DROPOUT = 0.1            # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 64          # Batch size
    NUM_EPOCHS = 20          # Number of training epochs
    LEARNING_RATE = 0.0001   # Learning rate
    
    # Device
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # ========================================
    # Create Dataset and DataLoader
    # ========================================
    print(f"\nCreating dataset...")
    print(f"  Samples: {NUM_SAMPLES}")
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    
    # Training and validation datasets
    train_dataset = ReverseDataset(NUM_SAMPLES, SEQ_LEN, VOCAB_SIZE)
    val_dataset = ReverseDataset(NUM_SAMPLES // 5, SEQ_LEN, VOCAB_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ========================================
    # Create Model
    # ========================================
    print(f"\nInitializing Transformer model...")
    print(f"  d_model: {D_MODEL}")
    print(f"  num_layers: {NUM_LAYERS}")
    print(f"  num_heads: {NUM_HEADS}")
    print(f"  d_ff: {D_FF}")
    
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=DROPOUT
    ).to(device)
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    
    # ========================================
    # Loss and Optimizer
    # ========================================
    # CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.PAD_IDX)
    
    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ========================================
    # Training Loop
    # ========================================
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        # Print results
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Val Acc:    {val_accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer.pth')
            print(f"  → Best model saved! ✓")
    
    # ========================================
    # Test the Model
    # ========================================
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Load best model
    model.load_state_dict(torch.load('best_transformer.pth'))
    
    # Test with examples
    test_model(model, val_dataset, device, num_examples=10)
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
