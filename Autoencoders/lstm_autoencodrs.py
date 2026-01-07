import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. LOAD REAL ECG DATA
# We use the ECG5000 dataset hosted publicly
url = "http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv"
dataframe = pd.read_csv(url, header=None)
raw_data = dataframe.values

# The last column is the label (0 or 1 in this specific version of the dataset)
# In this specific Google-hosted version: 1 = Normal, 0 = Anomaly
labels = raw_data[:, -1]
data = raw_data[:, 0:-1] 

train_data = data[labels == 1] # Train ONLY on Normal heartbeats
test_data = data[labels == 0]  # We will test on Anomalies later

# Normalize data to [0, 1]
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data) # Use same scaler

# Convert to PyTorch Tensors
# LSTM expects input shape: (Batch_Size, Sequence_Length, Features)
train_tensor = torch.FloatTensor(train_data).unsqueeze(2) 
test_tensor = torch.FloatTensor(test_data).unsqueeze(2)

train_loader = torch.utils.data.DataLoader(train_tensor, batch_size=64, shuffle=True)

# 2. DEFINE LSTM MODEL
class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(
            input_size=n_features, 
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=2,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # 1. ENCODE
        # hidden shape: (num_layers, batch, hidden_dim) -> (2, 64, 64)
        _, (hidden, _) = self.encoder(x)
        
        # 2. EXTRACT BOTTLENECK
        # We want the hidden state of the LAST layer only.
        # This represents the final summary of the sequence.
        # Shape becomes: (Batch, Hidden_Dim) -> (64, 64)
        hidden_last_layer = hidden[-1] 
        
        # 3. REPEAT
        # We repeat that single vector 140 times to create the input for the decoder
        # Shape becomes: (Batch, Seq_Len, Hidden_Dim) -> (64, 140, 64)
        decoder_input = hidden_last_layer.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # 4. DECODE
        decoded, _ = self.decoder(decoder_input)
        
        # 5. MAP TO OUTPUT
        return self.output_layer(decoded)

# 3. TRAIN
device = torch.device("mps" if torch.mps.is_available() else "cpu")
model = LSTMAutoencoder(seq_len=140, n_features=1, hidden_dim=128).to(device) # seq_len is 140 points
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction='mean')

print("Training on Normal Heartbeats...")
history = []
num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    train_losses = []
    for seq_true in train_loader:
        seq_true = seq_true.to(device)
        optimizer.zero_grad()
        
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true)
        
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    avg_loss = np.mean(train_losses)
    print(f'Epoch {epoch+1}: Loss {avg_loss:.4f}')

# 4. VISUALIZE: Normal vs Anomaly
# We take one Normal beat (from train) and one Anomaly (from test)
normal_sample = train_tensor[0].unsqueeze(0).to(device)
anomaly_sample = test_tensor[0].unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    pred_normal = model(normal_sample)
    pred_anomaly = model(anomaly_sample)

plt.figure(figsize=(12, 5))

# Plot Normal Reconstruction
plt.subplot(1, 2, 1)
plt.plot(normal_sample[0].cpu(), 'b', label='Input (Normal)')
plt.plot(pred_normal[0].cpu(), 'r', label='Reconstruction')
plt.fill_between(np.arange(140), normal_sample[0].cpu().flatten(), pred_normal[0].cpu().flatten(), color='lightcoral', alpha=0.5)
plt.title("Normal Heartbeat (Low Error)")
plt.legend()

# Plot Anomaly Reconstruction
plt.subplot(1, 2, 2)
plt.plot(anomaly_sample[0].cpu(), 'b', label='Input (Anomaly)')
plt.plot(pred_anomaly[0].cpu(), 'r', label='Reconstruction')
plt.fill_between(np.arange(140), anomaly_sample[0].cpu().flatten(), pred_anomaly[0].cpu().flatten(), color='lightcoral', alpha=0.5)
plt.title("Anomalous Heartbeat (High Error)")
plt.legend()

plt.show()