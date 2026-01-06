import re
import math
import random
import string
import numpy as np
import pandas as pd

from collections import Counter
from tabulate import tabulate

import sacrebleu

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import spacy

spacy_en = spacy.load("en_core_web_sm")
spacy_fr = spacy.load("fr_core_news_sm")

# Load English to French Dataset - I ran this in Kaggle, so yes.
data = pd.read_csv("/kaggle/input/eng-fra/eng-fra.txt", sep="\t", header=None).to_numpy()

# print(tabulate(data.head().to_numpy(), tablefmt="grid", headers=["English", "French"]))

def clean_text(text_data, rem_pun=True):
    # to lower case
    text_data = text_data.lower()
    
    # remove all html tags
    CLEANR = re.compile('<.*?>')
    text_data = re.sub(CLEANR, '', text_data)
    
    # remove punctuation
    if rem_pun:
        text_data = text_data.translate(str.maketrans('', '', string.punctuation))
        
    # remove extra white spaces
    text_data = re.sub(' +', ' ', text_data)
    
    # remove \u202f
    text_data = text_data.replace('\u202f', '')
        
    return text_data

data[:, 0] = [clean_text(text) for text in data[:, 0]]
data[:, 1] = [clean_text(text) for text in data[:, 1]]

def tokenize_eng(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_fra(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

tokenized_eng = [tokenize_eng(text) for text in data[:, 0].tolist()]
tokenized_fra = [tokenize_fra(text) for text in data[:, 1].tolist()]

VOCAB_SIZE = 10000

eng2idx = dict()
idx2eng = dict()

fra2idx = dict()
idx2fra = dict()

eng_count = Counter()
fra_count = Counter()

for text in tokenized_eng:
    eng_count.update(text)
    
for text in tokenized_fra:
    fra_count.update(text)

eng_common_words = eng_count.most_common(VOCAB_SIZE)
fra_common_words = fra_count.most_common(VOCAB_SIZE)

special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]

eng_tokens = special_tokens + sorted([i[0] for i in eng_common_words])
fra_tokens = special_tokens + sorted([i[0] for i in fra_common_words])

# Create token to index dictionaries
eng2idx = {token: idx for idx, token in enumerate(eng_tokens)}
idx2eng = {idx: token for idx, token in enumerate(eng_tokens)}
idx2fra = {idx: token for idx, token in enumerate(fra_tokens)}
fra2idx = {token: idx for idx, token in enumerate(fra_tokens)}

def encode_text(text_data, token_dict):
    text_data = [token_dict["<SOS>"]] + [token_dict[i] if i in token_dict.keys() else token_dict["<UNK>"] for i in text_data] + [token_dict["<EOS>"]]
    return text_data

def pad_or_truncate_keep_eos(seq, max_len=256, pad_token=0, eos_token=None):
    if len(seq) > max_len:
        if eos_token is None:
            return seq[:max_len]
        return seq[:max_len-1] + [eos_token]
    return seq + [pad_token] * (max_len - len(seq))


eng_dataset = [pad_or_truncate_keep_eos(encode_text(t, eng2idx), max_len=256, pad_token=eng2idx["<PAD>"], eos_token=eng2idx["<EOS>"])
               for t in tokenized_eng]

fra_dataset = [pad_or_truncate_keep_eos(encode_text(t, fra2idx), max_len=256, pad_token=fra2idx["<PAD>"], eos_token=fra2idx["<EOS>"])
               for t in tokenized_fra]

PAD_IDX = eng2idx["<PAD>"]
FRA_PAD_IDX = fra2idx["<PAD>"]
SOS_IDX = fra2idx["<SOS>"]
EOS_IDX = fra2idx["<EOS>"]
ENG_EOS_IDX = eng2idx["<EOS>"]



class ENG2FRA(Dataset):
    
    def __init__(self, eng_data, fra_data):
        self.eng_data = eng_data
        self.fra_data = fra_data
        
    def __len__(self):
        return len(self.eng_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        eng_text = torch.tensor(self.eng_data[idx], dtype=torch.long)
        fra_text = torch.tensor(self.fra_data[idx], dtype=torch.long)

        # Calculate actual lengths (excluding padding)
        eng_len = (eng_text != PAD_IDX).sum().item()  # Count non-pad tokens
        fra_len = (fra_text != FRA_PAD_IDX).sum().item()  # Count non-pad tokens

        return eng_text, fra_text, eng_len, fra_len
            
eng2fra_dataset = ENG2FRA(eng_data=eng_dataset, fra_data=fra_dataset)
eng2fra_loader = DataLoader(eng2fra_dataset, batch_size=128, shuffle=True)

class Encoder(nn.Module):
    def __init__(self, src_vocab, emb_dim=128, hid_dim=256, n_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(src_vocab, emb_dim, padding_idx=eng2idx["<PAD>"])
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.hid_dim = hid_dim
        self.n_layers = n_layers

    def forward(self, src, src_len):
        # src: [B, S], src_len: [B]
        device = src.device
        emb = self.dropout(self.emb(src))  # [B,S,E]

        # For MPS: avoid pack_padded_sequence by using regular LSTM forward
        # This is a workaround for MPS backend bugs with packed sequences
        if device.type == 'mps':
            # Regular forward without packing (slower but works on MPS)
            enc_out, (h, c) = self.lstm(emb)
        else:
            # Use packed sequences for efficiency on CPU/CUDA
            packed = pack_padded_sequence(emb, src_len.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h, c) = self.lstm(packed)
            enc_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        return enc_out, h, c


class Decoder(nn.Module):
    def __init__(self, tgt_vocab, emb_dim=128, hid_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(tgt_vocab, emb_dim, padding_idx=fra2idx["<PAD>"])
        self.lstm = nn.LSTM(
            emb_dim, hid_dim, num_layers=n_layers,
            batch_first=True, dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False
        )
        self.fc = nn.Linear(hid_dim, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h, c):
        # x: [B] (one token step) OR [B,1]
        if x.dim() == 1:
            x = x.unsqueeze(1)  # [B,1]

        emb = self.dropout(self.emb(x))  # [B,1,E]
        out, (h, c) = self.lstm(emb, (h, c))  # out: [B,1,H]
        logits = self.fc(out).squeeze(1)      # [B,V]
        return logits, h, c
        
    
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim=128, hid_dim=256, n_layers=2, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_dim, hid_dim, n_layers, dropout, bidirectional=True)
        self.decoder = Decoder(tgt_vocab, emb_dim, hid_dim, n_layers, dropout)

        # Use ONE bridge per state (works on tensors with shape [n_layers, B, 2H])
        self.bridge_h = nn.Linear(hid_dim * 2, hid_dim)
        self.bridge_c = nn.Linear(hid_dim * 2, hid_dim)

        self.n_layers = n_layers
        self.hid_dim = hid_dim
        self.tgt_vocab = tgt_vocab

    def _bridge(self, h, c):
        # h,c: [n_layers*2, B, H] -> [n_layers, B, 2H] -> [n_layers, B, H]
        B = h.size(1)
        H = h.size(2)

        h = h.contiguous().view(self.n_layers, 2, B, H)
        c = c.contiguous().view(self.n_layers, 2, B, H)

        h_cat = torch.cat((h[:, 0], h[:, 1]), dim=-1)  # [n_layers, B, 2H]
        c_cat = torch.cat((c[:, 0], c[:, 1]), dim=-1)

        h0 = torch.tanh(self.bridge_h(h_cat))          # [n_layers, B, H]
        c0 = torch.tanh(self.bridge_c(c_cat))
        return h0, c0

    def forward(self, src, src_len, tgt, teacher_forcing_ratio=0.5):
        B, T = tgt.shape
        V = self.tgt_vocab

        _, h, c = self.encoder(src, src_len)
        h, c = self._bridge(h, c)

        x = tgt[:, 0]  # <SOS>
        step_logits = []

        for t in range(1, T):
            logits, h, c = self.decoder(x, h, c)          # [B, V]
            step_logits.append(logits.unsqueeze(1))       # [B, 1, V]

            use_tf = (torch.rand((), device=tgt.device).item() < teacher_forcing_ratio)
            x = tgt[:, t] if use_tf else logits.argmax(dim=-1)

        return torch.cat(step_logits, dim=1)              # [B, T-1, V]


def sequence_ce_loss(logits, targets, pad_idx):
    """
    logits: [B, T, V]
    targets: [B, T]
    """
    B, T, V = logits.shape
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)
    return loss_fn(logits.reshape(B*T, V), targets.reshape(B*T))

def train_one_epoch(model, loader, optimizer, device, tf_ratio=0.5, clip=1.0):
    model.train()
    total_loss = 0.0

    for batch_idx, (src, tgt, src_len, tgt_len) in enumerate(loader):
        src, tgt = src.to(device), tgt.to(device)
        src_len = src_len.to(device)

        optimizer.zero_grad()
        logits = model(src, src_len, tgt, teacher_forcing_ratio=tf_ratio)  # [B,T-1,V]
        loss = sequence_ce_loss(logits, tgt[:, 1:], pad_idx=FRA_PAD_IDX)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if batch_idx % 400 == 0:
            print(f"Batch IDX: {batch_idx} \t|\t Loss: {loss.item():.4f}")

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_ppl(model, loader, device):
    model.eval()
    total_loss = 0.0

    for src, tgt, src_len, tgt_len in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_len = src_len.to(device)

        logits = model(src, src_len, tgt, teacher_forcing_ratio=0.0)
        loss = sequence_ce_loss(logits, tgt[:, 1:], pad_idx=FRA_PAD_IDX)
        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, math.exp(avg_loss)

@torch.no_grad()
def greedy_decode(model, src, src_len, max_len=60):
    model.eval()
    device = src.device

    # Encode
    _, h, c = model.encoder(src, src_len)
    h, c = model._bridge(h, c)

    # Start
    x = torch.full((src.size(0),), SOS_IDX, dtype=torch.long, device=device)

    out_tokens = []
    for _ in range(max_len):
        logits, h, c = model.decoder(x, h, c)
        x = logits.argmax(dim=-1)
        out_tokens.append(x)

    out_tokens = torch.stack(out_tokens, dim=1)  # [B, max_len]
    return out_tokens

def ids_to_tokens(ids, idx2tok, eos_idx, pad_idx):
    tokens = []
    for i in ids:
        if i == eos_idx:
            break
        if i == pad_idx:
            continue
        tok = idx2tok.get(int(i), "<UNK>")
        # optional: skip special tokens
        if tok not in ["<SOS>", "<EOS>", "<PAD>"]:
            tokens.append(tok)
    return tokens

class BeamNode:
    def __init__(self, sequence, log_prob, h, c):
        self.sequence = sequence  # List of token IDs so far
        self.log_prob = log_prob  # Cumulative log probability
        self.h = h                # Hidden state at this step
        self.c = c                # Cell state at this step

    def __lt__(self, other):
        # This allows us to sort nodes by score easily
        return self.log_prob < other.log_prob

@torch.no_grad()
def beam_search_decode(model, src, src_len, beam_width=3, max_len=60):
    """
    Decodes a SINGLE sentence using Beam Search.
    src: [1, seq_len] (Batch size must be 1)
    """
    model.eval()
    device = src.device

    # 1. ENCODE
    # We run the encoder once.
    enc_out, h, c = model.encoder(src, src_len)
    h, c = model._bridge(h, c)

    # 2. INITIALIZE BEAM
    # Start with just the <SOS> token
    # Log prob is 0.0 because log(1.0) = 0
    start_node = BeamNode(
        sequence=[SOS_IDX], 
        log_prob=0.0, 
        h=h, 
        c=c
    )
    
    # Our set of active candidates
    beams = [start_node]
    
    # A list to store completed sentences (those that hit <EOS>)
    completed_beams = []

    # 3. BEAM LOOP
    for _ in range(max_len):
        new_candidates = []

        # Expand every current beam
        for node in beams:
            # If this beam already finished, don't expand it further
            # (Ideally this shouldn't happen if logic below is correct, but safe check)
            if node.sequence[-1] == EOS_IDX:
                completed_beams.append(node)
                continue

            # Prepare input for decoder (the last word of this beam)
            # x shape: [1] -> [1, 1] inside decoder
            x = torch.tensor([node.sequence[-1]], dtype=torch.long, device=device)
            
            # Predict ONE step
            # h, c shape: [Layers, 1, Hid]
            logits, h_new, c_new = model.decoder(x, node.h, node.c)
            
            # Get Log Softmax (better numerical stability for adding probabilities)
            log_probs = F.log_softmax(logits, dim=-1) # [1, Vocab]

            # Get top K most likely next words
            # top_val: log probabilities, top_idx: token indices
            top_val, top_idx = log_probs.topk(beam_width, dim=-1)

            # Create new candidates
            for i in range(beam_width):
                token = top_idx[0][i].item()
                score = top_val[0][i].item()
                
                # Cumulative score = Old Score + New Log Prob
                new_node = BeamNode(
                    sequence = node.sequence + [token],
                    log_prob = node.log_prob + score,
                    h = h_new,
                    c = c_new
                )
                
                # Check if we just generated <EOS>
                if token == EOS_IDX:
                    completed_beams.append(new_node)
                else:
                    new_candidates.append(new_node)

        # 4. PRUNE (KEEP TOP K)
        # Sort by score (highest first)
        # Note: sorted is ascending, so we use reverse=True
        sorted_candidates = sorted(new_candidates, key=lambda x: x.log_prob, reverse=True)
        
        # Keep only the top 'beam_width' alive
        # We subtract len(completed_beams) to ensure we don't over-generate 
        # if we already found some good finished sentences.
        beams = sorted_candidates[:beam_width]

        # Stop if we have no active beams left (all hit EOS or died)
        if not beams:
            break
            
        # Stop early if we have enough completed sentences that are better than
        # any current partial sentence (Advanced optimization, optional)
        if len(completed_beams) >= beam_width:
             # Just a heuristic break to save time
             break

    # 5. FINALIZE
    # If we didn't finish any sentence (rare), treat current beams as finished
    if not completed_beams:
        completed_beams = beams

    # Sort completed beams by score
    completed_beams = sorted(completed_beams, key=lambda x: x.log_prob, reverse=True)

    # Return the single best sequence
    best_beam = completed_beams[0]
    
    # Convert list to tensor (compatible with your existing evaluation code)
    return torch.tensor(best_beam.sequence, dtype=torch.long, device=device).unsqueeze(0)

@torch.no_grad()
def compute_bleu(model, loader, device, max_len=60):
    model.eval()
    hyps = []
    refs = []

    for src, tgt, src_len, tgt_len in loader:
        src, tgt = src.to(device), tgt.to(device)
        src_len = src_len.to(device)

        pred = greedy_decode(model, src, src_len, max_len=max_len)  # [B,L]

        for b in range(src.size(0)):
            hyp_toks = ids_to_tokens(pred[b].cpu().tolist(), idx2fra, EOS_IDX, FRA_PAD_IDX)
            ref_toks = ids_to_tokens(tgt[b, 1:].cpu().tolist(), idx2fra, EOS_IDX, FRA_PAD_IDX)

            hyps.append(" ".join(hyp_toks))
            refs.append(" ".join(ref_toks))

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    return bleu


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(dataset, batch_size=128, seed=42, device=None):
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=generator)

    pin = device is not None and device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin)
    return train_loader, val_loader, test_loader


def print_sample_translations(model, loader, device, num_samples=3, max_len=60):
    model.eval()
    batch = next(iter(loader))
    src, tgt, src_len, _ = batch
    src, tgt, src_len = src.to(device), tgt.to(device), src_len.to(device)

    preds = greedy_decode(model, src, src_len, max_len=max_len)

    for i in range(min(num_samples, src.size(0))):
        src_toks = ids_to_tokens(src[i].cpu().tolist(), idx2eng, ENG_EOS_IDX, PAD_IDX)
        ref_toks = ids_to_tokens(tgt[i].cpu().tolist(), idx2fra, EOS_IDX, FRA_PAD_IDX)
        pred_toks = ids_to_tokens(preds[i].cpu().tolist(), idx2fra, EOS_IDX, FRA_PAD_IDX)

        print(f"--- Sample {i+1} ---")
        print(f"Src: {' '.join(src_toks)}")
        print(f"Ref: {' '.join(ref_toks)}")
        print(f"Hyp: {' '.join(pred_toks)}")

def print_beam_samples(model, loader, device, num_samples=3):
    model.eval()
    batch = next(iter(loader))
    src, tgt, src_len, _ = batch
    src, tgt, src_len = src.to(device), tgt.to(device), src_len.to(device)

    print(f"--- BEAM SEARCH (Width=3) ---")

    for i in range(min(num_samples, src.size(0))):
        # Slice inputs to get batch size = 1
        curr_src = src[i].unsqueeze(0)      # [1, seq_len]
        curr_len = src_len[i].unsqueeze(0)  # [1]
        
        # Run Beam Search
        pred_seq = beam_search_decode(model, curr_src, curr_len, beam_width=3)
        
        # Convert IDs to Words
        src_toks = ids_to_tokens(src[i].cpu().tolist(), idx2eng, ENG_EOS_IDX, PAD_IDX)
        ref_toks = ids_to_tokens(tgt[i].cpu().tolist(), idx2fra, EOS_IDX, FRA_PAD_IDX)
        pred_toks = ids_to_tokens(pred_seq[0].cpu().tolist(), idx2fra, EOS_IDX, FRA_PAD_IDX)

        print(f"Sample {i+1}")
        print(f"Src: {' '.join(src_toks)}")
        print(f"Ref: {' '.join(ref_toks)}")
        print(f"Hyp: {' '.join(pred_toks)}")
        print("-" * 20)


def run_training(epochs=10, batch_size=32, lr=1e-3, tf_start=1.0, tf_floor=0.3, clip=1.0, seed=42):
    set_seed(seed)
    device = torch.device("cuda")

    train_loader, val_loader, test_loader = make_dataloaders(eng2fra_dataset, batch_size=batch_size, seed=seed, device=device)

    model = Seq2Seq(len(eng_tokens), len(fra_tokens), emb_dim=256, hid_dim=512, n_layers=2, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_bleu = -float("inf")
    ckpt_path = "seq2seq_best.pt"

    for epoch in range(epochs):
        tf_ratio = max(tf_floor, tf_start - (epoch * 0.1))
        print(f"\n\nEpoch: {epoch}")
        print("-------" * 6)
        train_loss = train_one_epoch(model, train_loader, optimizer, device, tf_ratio=tf_ratio, clip=clip)
        print("\n\n")
        print("-------" * 6)
        val_loss, val_ppl = evaluate_ppl(model, val_loader, device)
        print("\n\n")
        print("-------" * 6)
        val_bleu = compute_bleu(model, val_loader, device)

        print(f"Epoch {epoch+1}/{epochs} | tf={tf_ratio:.2f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | val_bleu={val_bleu:.2f}")

        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_bleu": val_bleu,
            }, ckpt_path)
            print(f"Saved new best checkpoint with BLEU {val_bleu:.2f} to {ckpt_path}")

    # Load best and evaluate on test
    if best_bleu > -float("inf"):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])

    test_loss, test_ppl = evaluate_ppl(model, test_loader, device)
    test_bleu = compute_bleu(model, test_loader, device)
    print(f"Test | loss={test_loss:.4f} | ppl={test_ppl:.2f} | bleu={test_bleu:.2f}")

    print_sample_translations(model, test_loader, device)
    
    
# Currently we are just sampling the top probabilities (greedy argmax), but there are more techniques:
# 1. Beam Search
# 2. Temperature Scaling (We just add a value T, where we are effectively dividing the logits by T before exponentiating the e^(prediction_logits / T))
#   Where T < 1 sharpens (makes the higher probabilities the highest and rest all lowest)
#   then T > 1 flattens (makes the higher probabilities slightly not dominate the rest of the logits)
# 2. Top-p sampling
# 3. Top-k sampling


if __name__ == "__main__":
    run_training()