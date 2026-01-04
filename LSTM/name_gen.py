import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

np.random.seed(42)

baby_names = pd.read_csv("baby_names.csv", low_memory=False).to_numpy()
baby_names = baby_names[:, 1].reshape(-1, 1)

def unique_chars(baby_names_list):
    unq_set = set()
    max_len = -np.inf
    for i in baby_names_list:
        bn_set = set(i[0].lower())
        max_len = len(i[0]) if len(i[0]) > max_len else max_len
        unq_set = unq_set.union(bn_set)
        
    return list(unq_set), max_len

baby_names = baby_names.tolist()
unq_chars, max_len = unique_chars(baby_names)

tokens = ["<PAD>", "<SOS>", "<EOS>"]
tokens.extend(unq_chars)

char2idx = {char: idx for idx, char in enumerate(tokens)}
idx2char = {idx: char for idx, char in enumerate(tokens)}

def tokenize_names(name):
    name = name[0].lower()
    pad_len = max_len - len(name)
    tokenized_name = np.array([char2idx["<SOS>"]])
    for i in name:
        tokenized_name = np.append(tokenized_name, char2idx[i])
    tokenized_name = np.append(tokenized_name, [char2idx["<EOS>"]])
    tokenized_name = np.append(tokenized_name, [char2idx["<PAD>"]] * pad_len)
    return tokenized_name

tokenized_names = np.array(list(map(tokenize_names, baby_names)))
# print(tokenized_names.shape)
# print(tokenized_names[:5])

class NameLoader(torch.utils.data.Dataset):
    def __init__(self, tokenized_names):
        self.tokenized_names = tokenized_names

    def __len__(self):
        return len(self.tokenized_names)

    def __getitem__(self, index):
        seq = self.tokenized_names[index]  # shape: (max_len + 2,)

        pad_id = char2idx["<PAD>"]
        nonpad_len = len(seq) - np.count_nonzero(seq == pad_id)   # includes <SOS> and <EOS>
        actual_length = nonpad_len - 1  # because x = seq[:-1] has one fewer step

        x = torch.tensor(seq[:-1], dtype=torch.long)  # <SOS> ... last_token_removed
        y = torch.tensor(seq[1:],  dtype=torch.long)  # ... shifted by 1 (predict next token)

        return x, y, torch.tensor(actual_length, dtype=torch.long)

        
name_loader = NameLoader(tokenized_names)
train_dataset = torch.utils.data.DataLoader(name_loader, batch_size=128, collate_fn=None)

# test_data, test_target_data, actual_length = next(iter(train_dataset))

# print(test_data)
# print(test_target_data)
# print(actual_length)

class LSTM(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.emb = nn.Embedding(len(tokens), 128)
        self.lstm = nn.LSTM(
            input_size=128, hidden_size=256, num_layers=2, 
            batch_first=True, dropout=0.1, bidirectional=False
        )
        self.lin = nn.Linear(256, len(tokens))
        
        
    def forward(self, X, actual_length):
        X = self.emb(X)                    # (B, T, E)
        T = X.size(1)
        lengths = actual_length.cpu()      # safest for pack on MPS/CPU

        X_packed = torch.nn.utils.rnn.pack_padded_sequence(
            X, lengths=lengths, batch_first=True, enforce_sorted=False
        )

        out_packed, _ = self.lstm(X_packed)

        X_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(
            out_packed, batch_first=True, total_length=T
        )

        return self.lin(X_unpacked)        # (B, T, vocab)
        
        
device = "mps" if torch.mps.is_available() else "cpu"
model = LSTM()
model.to(device=device)

# Number of Epochs
epochs = 25

# Loss
cross_loss = nn.CrossEntropyLoss(ignore_index=char2idx["<PAD>"])

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optim, T_max=epochs
)

model.train()

train_loss_history = []
lr_history = []

for epoch in range(epochs):
    for batch_idx, (input_data, target_data, actual_length) in enumerate(train_dataset):
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        optim.zero_grad()
        y_pred = model.forward(input_data, actual_length)
        cal_loss = cross_loss(y_pred.view(-1, len(tokens)), target_data.view(-1))
        cal_loss.backward()
        train_loss_history.append(cal_loss.item())
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
    
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {cal_loss.item():.4f}")
    
    lr_history.append(optim.param_groups[0]["lr"])
    scheduler.step()
        

def plot_loss_curves(train_losses):
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Iteration (Batch Step)")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss Curve")
    plt.show()
    

@torch.no_grad()
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, forbid_ids=None):
    """
    logits: (V,) tensor
    returns: int token id
    """
    logits = logits.clone()

    # Forbid certain tokens (PAD, SOS, etc.)
    if forbid_ids:
        for tid in forbid_ids:
            logits[tid] = -float("inf")

    # Temperature
    temperature = max(1e-6, float(temperature))
    logits = logits / temperature

    # Top-k
    if top_k is not None and top_k > 0:
        top_k = min(int(top_k), logits.numel())
        kth_val = torch.topk(logits, top_k).values[-1]
        logits = torch.where(logits < kth_val, -float("inf"), logits)

    # Top-p (nucleus)
    if top_p is not None and 0.0 < top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)

        keep = cum <= top_p
        keep[0] = True  # always keep at least 1

        keep_idx = sorted_idx[keep]
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[keep_idx] = False
        logits[mask] = -float("inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate_name(model, max_new_chars=30, temperature=1.0, top_k=None, top_p=None, prefix=""):
    model.eval()
    
    sos_id = char2idx["<SOS>"]
    eos_id = char2idx["<EOS>"]
    pad_id = char2idx["<PAD>"]
    
    forbid = [pad_id, sos_id]
    
    ids = [sos_id]
    for ch in prefix.lower():
        if ch in char2idx:
            ids.append(char2idx[ch])
            
    hidden = None
    
    for tid in ids:
        x = torch.tensor([[tid]], device=device)
        emb = model.emb(x)
        out, hidden = model.lstm(emb, hidden)
        
    last_id = ids[-1]
    out_chars = []
    
    for _ in range(max_new_chars):
        x = torch.tensor([[last_id]], device=device)
        emb = model.emb(x)
        out, hidden = model.lstm(emb, hidden)
        logits = model.lin(out[0, 0])
        
        next_id = sample_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            forbid_ids=forbid
        )
        
        if next_id == eos_id:
            break

        out_chars.append(idx2char[next_id])
        last_id = next_id

    return "".join(out_chars)
    
    
@torch.no_grad()
def avg_logprob(model, name):
    model.eval()
    
    sos_id = char2idx["<SOS>"]
    eos_id = char2idx["<EOS>"]
    pad_id = char2idx["<PAD>"]
    
    ids = [sos_id]
    for ch in name.lower():
        if ch in char2idx:
            ids.append(char2idx[ch])
        else:
            return -1e9
    ids.append(eos_id)
    
    hidden = None
    total_lp = 0.0
    steps = 0
    
    for t in range(len(ids) - 1):
        x = torch.tensor([[ids[t]]], device=device)
        emb = model.emb(x)
        out, hidden = model.lstm(emb, hidden)
        logits = model.lin(out[0, 0])
        log_probs = F.log_softmax(logits, dim=-1)
        
        target = ids[t+1]
        if target == pad_id:
            continue
        total_lp += float(log_probs[target].item())
        steps += 1
    
    return total_lp / max(1, steps)


VOWELS = set(list("aeiouy"))

def pronounceable_heuristic(name, max_cons_run=3, max_vowel_run=3):
    if len(name) == 0:
        return False
    if not any(ch in VOWELS for ch in name):
        return False

    max_c = 0
    max_v = 0
    c_run = 0
    v_run = 0

    for ch in name.lower():
        if ch in VOWELS:
            v_run += 1
            c_run = 0
        else:
            c_run += 1
            v_run = 0
        max_c = max(max_c, c_run)
        max_v = max(max_v, v_run)

    if max_c > max_cons_run:
        return False
    if max_v > max_vowel_run:
        return False

    # Optional: avoid triple repeated letters
    for i in range(len(name) - 2):
        if name[i] == name[i+1] == name[i+2]:
            return False

    return True
    

def generate_and_evaluate(model, n=100, temperature=1.0, top_p=0.9, top_k=None):
    train_set = set([x[0].lower() for x in baby_names])
    
    rng = np.random.default_rng(42)
    sample_size = min(2000, len(baby_names))
    idxs = rng.choice(len(baby_names), size=sample_size, replace=False)
    real_scores = [avg_logprob(model, baby_names[i][0].lower()) for i in idxs]
    realism_threshold = float(np.percentile(real_scores, 5))
    
    generated = []
    for _ in range(n):
        nm = generate_name(model, temperature=temperature, top_p=top_p, top_k=top_k)
        generated.append(nm)
        
    # Metrics
    unique = len(set(generated))
    novel = sum((nm not in train_set) for nm in generated)
    pron = sum(pronounceable_heuristic(nm) for nm in generated)
    scores = [avg_logprob(model, nm) for nm in generated]
    realistic = sum(s >= realism_threshold for s in scores)

    print("=== Generation settings ===")
    print(f"temperature={temperature}, top_p={top_p}, top_k={top_k}")
    print()
    print("=== Summary metrics (out of 100) ===")
    print(f"Unique:        {unique}/100")
    print(f"Novel:         {novel}/100  (not in training set)")
    print(f"Pronounceable: {pron}/100  (heuristic)")
    print(f"Realistic:     {realistic}/100  (avg logprob >= real-name 5th percentile)")
    print()
    print(f"Realism threshold (5th percentile of real names): {realism_threshold:.3f}")
    print(f"Generated avg logprob: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")
    print()

    # Show a few examples with tags
    print("=== Sample outputs ===")
    for nm, sc in list(zip(generated, scores))[:20]:
        tags = []
        if nm in train_set: tags.append("SEEN")
        if pronounceable_heuristic(nm): tags.append("PRON")
        if sc >= realism_threshold: tags.append("REAL")
        print(f"{nm:20s}  score={sc:7.3f}  [{' '.join(tags)}]")

    return generated, scores, realism_threshold


plot_loss_curves(train_loss_history)
generated_names, gen_scores, thr = generate_and_evaluate(model, n=100, temperature=1.0, top_p=0.9)