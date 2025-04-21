import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from tqdm import tqdm

class KeyValueFeedForward(nn.Module):
    def __init__(self, input_dim, num_memories, output_dim):
        super().__init__()
        # keys and values as learnable parameters
        self.keys   = nn.Parameter(torch.randn(num_memories, input_dim))
        self.values = nn.Parameter(torch.randn(num_memories, output_dim))

    def forward(self, x):
        # x: (B, input_dim)
        attn = F.softmax(x @ self.keys.t(), dim=1)      # (B, M)
        out  = attn @ self.values                       # (B, output_dim)
        return out, attn

class MNISTKeyValueNet(nn.Module):
    def __init__(self, num_memories=64):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(28*28, 128)
        self.kv_ff   = KeyValueFeedForward(128, num_memories, 64)
        self.fc2     = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        h = F.relu(self.fc1(x))            # (B,128)
        kv_out, attn = self.kv_ff(h)       # (B,64), (B,M)
        logits = self.fc2(kv_out)          # (B,10)
        return logits, h, attn

epochs      = 5
batch_size  = 128
lr          = 1e-3
num_memories= 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
train_ds = datasets.MNIST(root='.', train=True, download=True,
                          transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

# Model
model = MNISTKeyValueNet(num_memories=num_memories).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=lr)
crit  = nn.CrossEntropyLoss()

# hidden activations (only from first epoch) (idk it didn't work otherwise)
all_hidden = []
all_labels = []

model.train()
for ep in range(1, epochs+1):
    total_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {ep}"):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits, h, _ = model(x)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        # collect only during first epoch
        if ep == 1:
            all_hidden.append(h.detach().cpu().numpy())
            all_labels.append(y.cpu().numpy())
    print(f"Epoch {ep} Loss: {total_loss/len(train_ds):.4f}")

# Stack activations and labels from first epoch
hs = np.concatenate(all_hidden, axis=0)  
ys = np.concatenate(all_labels, axis=0) 
print("Training complete; collected activations from first epoch.")

# Normalize hidden and keys
hs_norm = hs / np.linalg.norm(hs, axis=1, keepdims=True)
keys    = model.kv_ff.keys.detach().cpu()           
keys_norm = keys / keys.norm(dim=1, keepdim=True)

K = 50  # number of triggers 
triggers = []
for k in range(num_memories):
    sims = hs_norm @ keys_norm[k].numpy()            
    topk = np.argpartition(-sims, K)[:K]
    topk_sorted = topk[np.argsort(-sims[topk])]
    triggers.append({
        'key_index': k,
        'trigger_indices': topk_sorted,
        'trigger_labels': ys[topk_sorted].tolist()
    })
print(f"Extracted {K} triggers for each of {num_memories} keys from first epoch activations.")




#Key–Value Agreement
agreements = []
train_full = datasets.MNIST(root='.', train=True, download=False,
                            transform=transforms.ToTensor())
for rec in triggers:
    k   = rec['key_index']
    idxs= rec['trigger_indices']
    subset = Subset(train_full, idxs)
    loader = DataLoader(subset, batch_size=batch_size)
    n_corr = 0; n_tot = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits, h, attn = model(x)
        w_k = attn[:, k].unsqueeze(1)                     # (B,1)
        v_k = model.kv_ff.values[k].unsqueeze(0)          # (1,64)
        out_k = w_k * v_k                                 # (B,64)
        logits_k = model.fc2(out_k)                      # (B,10)
        preds = logits_k.argmax(dim=1)
        n_corr += (preds == y).sum().item()
        n_tot  += y.size(0)
    agreements.append({'key_index': k,
                       'agreement': n_corr / n_tot,
                       'n_triggers': n_tot})

df_agree = pd.DataFrame(agreements)
print(df_agree.head())


#Generate FFN Outputs (Dim / Layer Modes)
# Choose mode: 'dim' or 'layer'
mode = 'layer'
max_samples = 1000  # e.g., first 1000 test examples

test_ds = datasets.MNIST(root='.', train=False, download=True,
                         transform=transforms.ToTensor())
if max_samples:
    test_ds = Subset(test_ds, list(range(max_samples)))
loader = DataLoader(test_ds, batch_size=batch_size)

records = []
for sid, (x, y) in enumerate(loader):
    x, y = x.to(device), y.to(device)
    with torch.no_grad():
        logits, h, attn = model(x)
    B = x.size(0)
    for b in range(B):
        for k in range(num_memories):
            w_k = attn[b, k].item()
            v_k = model.kv_ff.values[k]
            if mode == 'dim':
                for d, vkd in enumerate(v_k):
                    records.append({
                        'sample': sid*batch_size + b,
                        'true_label': y[b].item(),
                        'key': k,
                        'dim': d,
                        'contribution': w_k * vkd.item()
                    })
            else:
                contrib = (w_k * v_k).sum().item()
                records.append({
                    'sample': sid*batch_size + b,
                    'true_label': y[b].item(),
                    'key': k,
                    'layer_contribution': contrib
                })

# Build DataFrame and show summary
df_ffn = pd.DataFrame.from_records(records)
print(df_ffn.head())