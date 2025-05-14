import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# simple with 3 layers
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)  
        self.fc2 = nn.Linear(128, 64)  
        self.fc3 = nn.Linear(64, 10)    
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x1 = self.relu(self.fc1(x))  
        x2 = self.relu(self.fc2(x1)) 
        x3 = self.fc3(x2)            
        return x3, x1, x2

# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

# Initialize
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output, _, _ = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output, _, _ = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)

    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

def interpret_as_key_value_memory(model):
    #extracting weights
    W1 = model.fc1.weight.data  # Shape: [128, 784]
    W2 = model.fc2.weight.data  # Shape: [64, 128]
    W3 = model.fc3.weight.data  # Shape: [10, 64]

    keys_layer1 = W1.t()  
    values_layer1 = W2.t()  

    keys_layer2 = W2.t()  
    values_layer2 = W3.t()  

    return keys_layer1, values_layer1, keys_layer2, values_layer2

def visualize_keys(keys_layer1):
    for i in range(min(8, keys_layer1.shape[1])):  #  first 8 keys
        key = keys_layer1[:, i].reshape(28, 28)
        plt.subplot(4, 4, i+1)
        plt.imshow(key.numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f'Key {i}')
    plt.tight_layout()
    plt.show()

def find_top_activating_images(model, data_loader, num_keys=5, top_k=5):
    all_activations = [[] for _ in range(num_keys)]
    all_images      = [[] for _ in range(num_keys)]

    with torch.no_grad():
        for data, _ in data_loader:
            # Get layer-1 activations
            _, x1, _ = model(data)    
            batch = data.cpu().numpy()  

            # Accumulate
            for key_idx in range(num_keys):
                acts = x1[:, key_idx].cpu().numpy()  # [batch]
                for act_val, img in zip(acts, batch):
                    all_activations[key_idx].append(act_val)
                    all_images[key_idx].append(img.squeeze())

    for key_idx in range(num_keys):
        acts_np = np.array(all_activations[key_idx])
        # get top_k indices
        top_inds = np.argsort(acts_np)[-top_k:][::-1]

        plt.figure(figsize=(15, 3))
        plt.suptitle(f'Top {top_k} activating images for Key {key_idx}')
        for i, idx in enumerate(top_inds):
            plt.subplot(1, top_k, i+1)
            plt.imshow(all_images[key_idx][idx], cmap='gray')
            plt.title(f'{acts_np[idx]:.2f}')
            plt.axis('off')
        plt.show()

def analyze_key_activations_by_digit(model, data_loader):
    key_activations_by_digit = {digit: [] for digit in range(10)}

    with torch.no_grad():
        for data, labels in data_loader:
            _, activations, _ = model(data)

            for i, label in enumerate(labels):
                digit = label.item()
                key_activations_by_digit[digit].append(activations[i].numpy())

    # Average activations for digit
    avg_activations = {}
    for digit, acts in key_activations_by_digit.items():
        avg_activations[digit] = np.mean(np.vstack(acts), axis=0)

    # Find top keys for each digit
    for digit, avg_act in avg_activations.items():
        top_keys = np.argsort(avg_act)[-5:]  # Top 5 most activated keys
        print(f"Digit {digit} is most strongly detected by keys: {top_keys}")
    
    #plot
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array([avg_activations[d] for d in range(10)]), aspect='auto', cmap='viridis')
    plt.colorbar(label='Average activation')
    plt.xlabel('Key index')
    plt.ylabel('Digit')
    plt.yticks(range(10))
    plt.title('Key activations by digit')
    plt.show()

def key_values_for_ith_kv_pair(model, i):
    keys = model[i]


def visualize_key_values_for_ith_kv_pair(model, i: int):
    keys = model



print("\nEvaluating model accuracy...")
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        outputs, _, _ = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

print("\nInterpreting the model as key-value memory...")
keys_layer1, values_layer1, keys_layer2, values_layer2 = interpret_as_key_value_memory(model)

print("\nVisualizing first layer keys")
visualize_keys(keys_layer1)

print("\nFinding images that strongly activate specific keys")
find_top_activating_images(model, test_loader, num_keys=1)



