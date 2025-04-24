import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# Define a simple network with 3 layers
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)  # First layer
        self.fc2 = nn.Linear(128, 64)   # Second layer
        self.fc3 = nn.Linear(64, 10)    # Output layer
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x1 = self.relu(self.fc1(x))  # Layer 1 output
        x2 = self.relu(self.fc2(x1)) # Layer 2 output
        x3 = self.fc3(x2)            # Final output
        return x3, x1, x2

def interpret_as_key_value_memory(model):
    # Extract weights from the model
    W1 = model.fc1.weight.data  # Shape: [128, 784]
    W2 = model.fc2.weight.data  # Shape: [64, 128]
    W3 = model.fc3.weight.data  # Shape: [10, 64]
    
    # Layer 1: Keys = W1.T, Values = W2.T
    keys_layer1 = W1.t()  # Shape: [784, 128]
    values_layer1 = W2.t()  # Shape: [128, 64]
    
    # Layer 2: Keys = W2.T, Values = W3.T
    keys_layer2 = W2.t()  # Shape: [128, 64]
    values_layer2 = W3.t()  # Shape: [64, 10]
    
    return keys_layer1, values_layer1, keys_layer2, values_layer2

def visualize_keys(keys_layer1):
    plt.figure(figsize=(12, 8))
    # Reshape first layer keys for visualization (each column is a key)
    for i in range(min(16, keys_layer1.shape[1])):  # Show first 16 keys
        key = keys_layer1[:, i].reshape(28, 28)
        plt.subplot(4, 4, i+1)
        plt.imshow(key.numpy(), cmap='viridis')
        plt.axis('off')
        plt.title(f'Key {i}')
    plt.tight_layout()
    plt.show()

def find_top_activating_images(model, data_loader, num_keys=5):
    # Find images that maximally activate specific keys
    with torch.no_grad():
        top_activations = [[] for _ in range(num_keys)]
        top_images = [[] for _ in range(num_keys)]
        top_labels = [[] for _ in range(num_keys)]
        
        for data, labels in data_loader:
            # Get middle activations
            _, x1, _ = model(data)
            
            # For the first few keys, find top activating images
            for key_idx in range(num_keys):
                activations = x1[:, key_idx]
                values, indices = torch.topk(activations, k=5)
                
                for val, idx in zip(values, indices):
                    top_activations[key_idx].append(val.item())
                    top_images[key_idx].append(data[idx].squeeze().numpy())
                    top_labels[key_idx].append(labels[idx].item())
        
        # Show top activating images for each key
        for key_idx in range(num_keys):
            plt.figure(figsize=(15, 3))
            plt.suptitle(f'Top activating images for Key {key_idx}')
            for i in range(5):
                plt.subplot(1, 5, i+1)
                plt.imshow(top_images[key_idx][i], cmap='gray')
                plt.title(f'Digit: {top_labels[key_idx][i]}, Act: {top_activations[key_idx][i]:.2f}')
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
    
    # Average activations for each digit
    avg_activations = {}
    for digit, acts in key_activations_by_digit.items():
        avg_activations[digit] = np.mean(np.vstack(acts), axis=0)
    
    # Find top keys for each digit
    for digit, avg_act in avg_activations.items():
        top_keys = np.argsort(avg_act)[-5:]  # Top 5 most activated keys
        print(f"Digit {digit} is most strongly detected by keys: {top_keys}")
    
    # Visualize how each digit activates different keys
    plt.figure(figsize=(12, 8))
    plt.imshow(np.array([avg_activations[d] for d in range(10)]), aspect='auto', cmap='viridis')
    plt.colorbar(label='Average activation')
    plt.xlabel('Key index')
    plt.ylabel('Digit')
    plt.yticks(range(10))
    plt.title('Key activations by digit')
    plt.show()
    
    # Identify keys that might detect circles
    circular_digits = [0, 6, 8, 9]
    non_circular_digits = [1, 7]
    
    # Average activation for circular vs non-circular digits
    circular_activation = np.mean([avg_activations[d] for d in circular_digits], axis=0)
    non_circular_activation = np.mean([avg_activations[d] for d in non_circular_digits], axis=0)
    
    # Find keys that respond much more to circular digits
    diff = circular_activation - non_circular_activation
    circle_detector_keys = np.argsort(diff)[-5:]
    
    print("\nPotential circle detector keys:", circle_detector_keys)
    return circle_detector_keys

# Main execution
def run_mnist_key_value_analysis():
    print("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)

    print("Creating and training the model...")
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(3):  # Just 3 epochs for demonstration
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output, _, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
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
    
    print("\nVisualizing first layer keys (patterns the model is looking for)...")
    visualize_keys(keys_layer1)
    
    print("\nFinding images that strongly activate specific keys...")
    find_top_activating_images(model, test_loader, num_keys=3)
    
    print("\nAnalyzing how different digits activate different keys...")
    circle_detector_keys = analyze_key_activations_by_digit(model, test_loader)
    
    # Demonstrate usage of circle detector keys
    print("\nDemonstrating circle detection:")
    with torch.no_grad():
        # Get a batch of test images
        for data, labels in test_loader:
            # Just use the first 10 images for demonstration
            sample_data = data[:10]
            sample_labels = labels[:10]
            
            # Get activations
            _, activations, _ = model(sample_data)
            
            # Check circle detector activations
            for i in range(10):
                circle_activation = np.mean([activations[i, key].item() for key in circle_detector_keys])
                has_circle = "Likely has circles" if circle_activation > 0.5 else "Likely no circles"
                print(f"Image {i}, Digit {sample_labels[i].item()}: {has_circle} (avg activation: {circle_activation:.2f})")
            
            break  # Just one batch

if __name__ == "__main__":
    run_mnist_key_value_analysis()