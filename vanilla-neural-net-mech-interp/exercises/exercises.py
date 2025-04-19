# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# %%

# hyper‑params
BATCH_SIZE   = 128
HIDDEN_DIM   = 256
EPOCHS       = 5
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data
transform = transforms.Compose([
    transforms.ToTensor(),                         # (1, 28, 28) in [0,1]
    transforms.Lambda(lambda t: t.view(-1))        # flatten to (784,)
])

train_set = datasets.MNIST(root="data", train=True,  download=True, transform=transform)
test_set  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

# model
class ThreeLayerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)   # 784 → hidden
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # hidden → 10

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)            # raw logits (CrossEntropyLoss adds softmax)

model = ThreeLayerNet(28*28, HIDDEN_DIM, 10).to(DEVICE)

def train_model(model: ThreeLayerNet):
  # loss & optim
  criterion  = nn.CrossEntropyLoss()
  optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
  final_accuracy = None

  # training loop
  for epoch in range(1, EPOCHS + 1):
      model.train()
      running_loss = 0.0
      for xb, yb in train_loader:
          xb, yb = xb.to(DEVICE), yb.to(DEVICE)
          logits = model(xb)
          loss   = criterion(logits, yb)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          running_loss += loss.item() * xb.size(0)

      avg_loss = running_loss / len(train_loader.dataset)

      # quick validation
      model.eval()
      correct = 0
      train_losses = []
      with torch.no_grad():
        for xb, yb in train_loader:
          xb, yb = xb.to(DEVICE), yb.to(DEVICE)
          logits = model(xb)
          loss   = criterion(logits, yb)
          train_losses.append(loss.item())
        final_train_loss = sum(train_losses) / len(train_losses)

      test_losses = []
      with torch.no_grad():
          for xb, yb in test_loader:
              xb, yb = xb.to(DEVICE), yb.to(DEVICE)
              logits = model(xb)
              loss = criterion(logits, yb)
              test_losses.append(loss.item())
              pred = logits.argmax(dim=1)
              correct += (pred == yb).sum().item()
      acc = correct / len(test_loader.dataset) * 100
      final_test_loss = sum(test_losses) / len(test_losses)
      final_accuracy = acc

      print(f"Epoch {epoch:2d} | loss: {avg_loss:.4f} | test acc: {acc:.2f}%")
  print(f"Final train loss: {final_train_loss} | final test loss: {final_test_loss} | final test accuracy: {final_accuracy}")

# %%

different_hidden_layer_sizes = [
    HIDDEN_DIM,
    2 * HIDDEN_DIM,
    3 * HIDDEN_DIM,
    4 * HIDDEN_DIM,
    5 * HIDDEN_DIM,
    6 * HIDDEN_DIM,
]

for hidden_layer_size in different_hidden_layer_sizes:
  train_model(ThreeLayerNet(28*28, hidden_layer_size, 10).to(DEVICE))

# %%

# Run this for different hidden layer sizes to look at double descent
#
# Hypotheses I would have:
#
# 1. It might be the case that we actually see very interpretable features show
# up once we get over the initial hump in double descent
# 2. We might not get interpretable features show up, we might end up with
# features that look similar to what happened before the interpolation
# threshold, but have a lot of "dead" neurons, that is neurons (under our
# key-value interpretation of a neuron) that have "detector vectors" with close
# to 0 magnitude.
#
# But either way we should be able to confirm things by simply plotting what the
# "detector" vectors look like (which should interpretable as simple 2-d
# diagrams) and then we can list off the "most important" neurons as those with
# "detector" vectors and "output" vectors that have the largest combined
# magnitude (for some definition of combined).

# %%