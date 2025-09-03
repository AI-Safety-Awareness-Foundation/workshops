# %% [markdown]
"""
This notebook is meant to be an interactive exploration of how we can use a
key-value perspective on a single hidden-layer neural net to get much more
interpretable results about how a basic neural net is able to do image
recognition of hand-written digits from the MNIST dataset.

This is meant to be used alongside a set of slides found in the associated
GitHub repository.

It's split into three main components. We'll first revisit
"""

# %%

# Some basic imports that will be necessary for any of our code to be able to
# run.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import einops

# %% [markdown]
"""
Let's begin with a very quick look at the single hidden layer neural net we're
using for our exercise today. This kind of basic neural net goes by many names,
whether it's an MLP (Multi-Layer Perceptron), feed-forward neural net, vanilla
neural net, etc.

A neural net with a single hidden layer is enough to approximate any function if
the net is large enough, so in theory could be used for many different tasks.

In practice, for many domains it turns out it is extremely difficult to actually
get a single hidden layer neural net to learn the relevant task at hand within a
feasible computational and data budget. However, it turns out to be enough to
get reasonable scores for recognizing hand-written digits in the MNIST dataset.

Moreover, it's one of the most basic components of modern machine learning
and AI that still exhibits enough complexity to be interesting and is often
treated as an opaque black box we do not understand.

This exploration will hopefully show you some ways of breaking open this black
box and making it more understandable!

For MNIST digit recognition, we'll set the input dimension to 784 (for 28x28
images) and the output dimension to 10 (one for each category of digit). But
we'll leave these flexible, as well as whether to use softmax or not, because
we'll use this same architecture to illustrate some of the tiny neural nets we
presented in the slides.

Spend a few minutes reviewing the architecture of `SimpleNN` to make sure you
understand what's going on. The default settings for `__init__` are all meant
for our main MNIST digit recognition task.
"""

# %%

# simple NN with 3 layers
class SimpleNN(nn.Module):
    def __init__(self, hidden_dim: int, input_dim=784, output_dim=10, has_bias: bool=True, use_softmax: bool=True):
        super(SimpleNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=has_bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=has_bias)
        self.relu = nn.ReLU()
        self.use_softmax = use_softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.flatten(x)
        x1 = self.relu(self.fc1(x))
        x2 = self.fc2(x1)
        if self.use_softmax:
          return self.softmax(x2)
        else:
           return x2

# %% [markdown]
"""
Okay now we want to reframe our model using the key-value visualization So it's
time to implement the functions that will pull out the ith key and ith value in
our neural net.

These functions are just one line each, but they are extremely important to
understand. As a reminder, the connections between two layers of a neural net
can be described as a matrix.

For example, in the following neural net where A and B, C and D, and E and F all
form respective neural net layers, this can be represented as two matrices
describing the connections from the AB layer to the CD layer and then from the
CD layer to the EF layer.

```
A -- C -- E
  \ /  \ /
  / \  / \ 
B -- D -- F
```

These two matrices look like

$\begin{bmatrix} AC & BC \\ AD & BD \end{bmatrix}$

and

$\begin{bmatrix} CE & DE \\ CF & DF \end{bmatrix}$

where e.g. $AC$ refers to the weight of the connection from $A$ to $C$, $BD$ to
the weight of the connection from $B$ to $D$, etc.

If we split the neural net into key-value pairs, we end up with the following:

```
A -- C -- E
    /  \  
  /      \ 
B         F
```

and 

```
A         E
  \      /
    \  /   
B -- D -- F
```
.

If you observe the key and value of the first key value pair, you'll find that
they are $\begin{bmatrix} AC \\ BC \end{bmatrix}$ and $\begin{bmatrix} CE \\ CF
\end{bmatrix}$ respectively. That is the first key corresponds to the first row
of the first matrix and the first value corresponds to the first column of the
second matrix.

This holds for the second key-value pair as well. More generally, the $i$-th
key-value pair will have its key correspond to the $i$-th row of the first matrix
and the $i$-th column of the second matrix.

The reason we alternate between rows and columns of the respective matrices is
that the rows of a matrix correspond to the weights of connections coming into a
neuron and the columns of a matrix correspond to the weights of all the
connections broadcasting out to the next set of neurons.

The traditional view of neural nets where each neuron takes $n$ inputs and gives
one output focuses primarily on this "per-row" perspective, but in our key-value
perspective we combine the "per-row" and "per-column" perspective.

This per-row and per-column perspective immediately leads to the one-line
implementations of `pull_out_ith_key` and `pull_out_ith_value`.
"""

# %%


def pull_out_ith_key(model, i):
  return model.fc1.weight[i]

def pull_out_ith_value(model, i):
  return model.fc2.weight[:, i]

# %% [markdown]
"""
This is all a lot of visualization code which you can either read or just run.

Unlike the previous section, you don't need to understand these too well.
"""

# %%

#plots the image
def visualize_image(image):
  norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
  plt.imshow(image.detach().numpy(), cmap='seismic', norm=norm)
  plt.axis('off')
  plt.show()

#plots a heatmap of a key
def visualize_ith_key(model, i, x_size=28, y_size=28):
  key = pull_out_ith_key(model, i).reshape(x_size, y_size)
  if model.fc1.bias is not None:
    key_bias = model.fc1.bias[i]
  else:
     key_bias = 0
  norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
  plt.imshow(key.detach().numpy(), cmap='seismic', norm=norm)
  plt.axis('off')
  plt.title(f'Key {i} (bias: {key_bias})')
  plt.show()

#visualizes a value
def visualize_ith_value(model, i):
  value = pull_out_ith_value(model, i).unsqueeze(0)
  norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
  plt.imshow(value.detach().numpy(), cmap='seismic', norm=norm)
  for x in range(value.shape[1]):
    plt.text(x, 0, f'{value[0, x].item():.3f}', ha='center', va='center', color='black', fontsize=6)
  plt.axis('off')
  plt.title(f'Value {i}')
  plt.show()

#visualizes the global value bias for each digit, or the baseline before any interactions
def visualize_value_bias(model):
  value = model.fc2.bias.unsqueeze(0)
  norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
  plt.imshow(value.detach().numpy(), cmap='seismic', norm=norm)
  for x in range(value.shape[1]):
    plt.text(x, 0, f'{value[0, x].item():.3f}', ha='center', va='center', color='black', fontsize=6)
  plt.axis('off')
  plt.title(f'Global value bias')
  plt.show()

#combines the above 3 visualization functions
def visualize_ith_key_value(model, i, key_x_size=28, key_y_size=28):
  visualize_ith_key(model, i, x_size=key_x_size, y_size=key_y_size)
  visualize_ith_value(model, i)
  if model.fc2.bias is not None:
    visualize_value_bias(model)

#Shows most influential interaction areas between an image and key 
def visualize_element_wise_multi_of_key_image(model, i, image, key_x_size=28, key_y_size=28):
  key = model.fc1.weight[i].reshape(key_x_size, key_y_size)
  element_wise_multi = key * image
  norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
  plt.imshow(element_wise_multi.detach().numpy(), cmap='seismic', norm=norm)
  plt.axis('off')
  plt.title(f'Element-wise multiplication of key {i} with image')
  plt.show()
  print(f"Dot-Product: {torch.sum(element_wise_multi)}")

#combines all of the above visualization functions
def visualize_ith_key_value_on_image(model, i, image, key_x_size=28, key_y_size=28):
  visualize_ith_key_value(model, i, key_x_size=key_x_size, key_y_size=key_y_size)
  visualize_element_wise_multi_of_key_image(model, i, image, key_x_size=key_x_size, key_y_size=key_y_size)

# %% [markdown]
"""
Let's play around a little with our visualization functions.

We can being by visualizing a few 2x2 images.

Instead of using black and white, as in our slides, we'll use two contrasting
colors in addition to white, since once we plot things that aren't just image
pixels, we may need to plot both positive and negative numbers. We'll use
increasingly darker shades of red to indicate positive numbers of increasing
magnitude, increasingly darker shades of blue to indicate negatives numbers of
increasing magnitude, and white to indicate 0.

This means most of our images of handwritten digits will consist of red and
white (red for 1 and white for 0), although our keys and value vectors will
consist of red, white, and blue.
"""

# %%

input_0 = torch.Tensor(
   [
      [1, 0],
      [1, 0],
   ]
)

visualize_image(input_0)

# %%

input_1 = torch.Tensor(
   [
      [1, 0],
      [1, 1],
   ]
)

visualize_image(input_1)

# %%

input_2 = torch.Tensor(
   [
      [0, 0],
      [1, 0],
   ]
)

visualize_image(input_2)

# %% [markdown]
"""
We now recreate the basic neural net we used in our slides to demonstrate the
key-value decomposition of a neural net.
"""

# %%

# When we first initialize a SimpleNN, all parameters 
example_nn = SimpleNN(hidden_dim=2, input_dim=4, output_dim=2)
preset_fc1 = nn.Parameter(
  torch.Tensor(
    [
      [1, 0, 0, 0],
      [0, 0, 0, 1],
    ]
  )
)

preset_fc2 = nn.Parameter(torch.Tensor(
   [
      [1, 0.5],
      [0, 0.5],
   ]
))
example_nn.fc2.weight = preset_fc2
print(f"{example_nn.fc2.weight=}")

# %%

# %%

example_nn(input_0)

# %%

visualize_ith_key_value(
  model=example_nn, 
  i=0, 
  key_x_size=2, 
  key_y_size=2
)

# %%

# None of this is code that you will need to write, but you should read this
# over to understand the structure of what kind of nets we'll be training.
#
# Note that we only train with 10,000 images out of the 60,000 image dataset!
# Originally this was because I was hoping to demonstrate some interesting
# double descent phenomena, but unfortunately I ran out of time to do that :(.
# Nonetheless, as we'll see, 10,000 images in the train set is actually enough
# to get to a very well trained neural net!

# hyper-params
BATCH_SIZE = 512
TRAIN_SET_SIZE = 10000
HIDDEN_DIM = 256
EPOCHS = 5000
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If you set this to True then this will train all the models from scratch,
# otherwise it will look for pre-saved weights and load those instead
TRAIN_FROM_SCRATCH = False
# When training, should we load the entire image set into GPU memory
LOAD_EVERYTHING_INTO_GPU_MEMORY = True

# %%


# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000)

hidden_dims = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
models = [SimpleNN(hidden_dim) for hidden_dim in hidden_dims]

# %%

# This is code that you can read if you'd like, but can also just run. It's
# mainly useful if you wanted to train these models yourself.

if LOAD_EVERYTHING_INTO_GPU_MEMORY:
  # We'll load into memory to make this faster
  train_loader_with_entire_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=train_dataset.data.shape[0])
  for batch_idx, (data, target) in enumerate(train_loader_with_entire_dataset):
      data = data[:TRAIN_SET_SIZE].to(DEVICE)
      target = torch.nn.functional.one_hot(target[:TRAIN_SET_SIZE], num_classes=10).to(torch.float)
      target = target.to(DEVICE)

  train_dataset = torch.utils.data.TensorDataset(data, target)
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

  test_loader_with_entire_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=test_dataset.data.shape[0])
  for test_data, test_target in test_loader_with_entire_dataset:
      test_data = test_data.to(DEVICE)
      test_target = torch.nn.functional.one_hot(test_target, num_classes=10).to(torch.float)
      test_target = test_target.to(DEVICE)


  test_dataset = torch.utils.data.TensorDataset(test_data, test_target)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000)

  # PyTorch DataLoader seems absurdly slow for MNIST dataset sizes
  # It seems to be calling get_item one by one instead of doing batch operations
  # Let's just do a custom list instead
  def generate_simple_loader(dataset, batch_size):
    permuted_indices = torch.randperm(dataset.tensors[0].shape[0])
    permuted_data = dataset.tensors[0][permuted_indices]
    permuted_target = dataset.tensors[1][permuted_indices]
    simple_loader = []
    for i in range(0, permuted_data.shape[0], batch_size):
      simple_loader.append((permuted_data[i:i+batch_size], permuted_target[i:i+batch_size]))
    return simple_loader

  simple_train_loader = generate_simple_loader(train_dataset, BATCH_SIZE)
  simple_test_loader = generate_simple_loader(test_dataset, 10000)

  train_loader = simple_train_loader
  test_loader = simple_test_loader

# %%

# This is the actual training loop! Even though this is not code you will need
# to write, you should definitely read this! It's good to understand exactly how
# our model is being trained!
#
# You might notice that we're using MSELoss instead of cross-entropy loss. It
# turns out that this is enough to get quite reasonable models and considerably
# simplifies some of the presentataion to people who have only an introductory
# understanding of neural nets.

train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
for model in models:
    print(f"Processing hidden_dim {model.hidden_dim}")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # Training
    train_loss = 0
    train_accuracy = 0
    train_samples = 0
    if TRAIN_FROM_SCRATCH:
      model = model.to(DEVICE)
      for epoch in range(EPOCHS):
          if LOAD_EVERYTHING_INTO_GPU_MEMORY:
              # Re-shuffle the train loader
              train_loader = generate_simple_loader(train_dataset, BATCH_SIZE)
          for batch_idx, (data, target) in enumerate(train_loader):
              optimizer.zero_grad()
              output = model(data)
              loss = criterion(output, target)
              loss.backward()
              optimizer.step()
    else:
      model.load_state_dict(torch.load(f"mnist_model_hidden_layer_{model.hidden_dim}"))
      model = model.to(DEVICE)
    with torch.no_grad():
      for data, target in train_loader:
          output = model(data)
          train_loss += criterion(output, target).item()
          train_accuracy += (output.argmax(dim=1) == target.argmax(dim=1)).sum().item()
          train_samples += data.shape[0]
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy / train_samples)

    # Testing
    test_loss = 0
    test_accuracy = 0
    test_samples = 0
    with torch.no_grad():
        for test_data, test_target in test_loader:
            output = model(test_data)
            loss = criterion(output, test_target)
            test_loss += loss.item()
            test_accuracy += (output.argmax(dim=1) == test_target.argmax(dim=1)).sum().item()
            test_samples += test_data.shape[0]
    test_losses.append(test_loss / len(test_loader))
    test_accuracies.append(test_accuracy / test_samples)

plt.plot(hidden_dims, train_losses, label='Train Loss')
plt.plot(hidden_dims, test_losses, label='Test Loss')
plt.xlabel('Hidden Dimension')
plt.ylabel('Loss')
plt.title('Loss vs Hidden Dim')
plt.xscale("log")
plt.legend()
plt.show()

plt.plot(hidden_dims, train_accuracies, label='Train Accuracy')
plt.plot(hidden_dims, test_accuracies, label='Test Accuracy')
plt.xlabel('Hidden Dimension')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Hidden Dim')
plt.xscale("log")
plt.legend()
plt.show()

# %%

if TRAIN_FROM_SCRATCH:
  for dim, model in zip(hidden_dims, models):
    # Save on CPU because this makes it easier to load for more devices
    model = model.to("cpu")
    torch.save(model.state_dict(), f"mnist_model_hidden_layer_{dim}")

# %%

# Go ahead and run this just to make sure

for model in models:
  model = model.to(DEVICE)

# %%

# calculates the accuracy of the model for each digit.

def accuracy_by_digit(model, loader):
  correct = [0] * 10
  total = [0] * 10
  with torch.no_grad():
    for data, target_probs in loader:
      output_probs = model(data)
      output = output_probs.argmax(dim=1)
      target = target_probs.argmax(dim=1)
      for i in range(target.shape[0]):
        total[target[i]] += 1
        if output[i] == target[i]:
          correct[target[i]] += 1
  return [correct[i] / total[i] for i in range(10)]

# %%

# Test it out for our 131072 hidden units model
accuracy_by_digit(models[14], test_loader)

# %%

# Test it out for our 8 hidden units model
accuracy_by_digit(models[0], test_loader)

# %%

# Now that we have a way of pulling out keys and values, we can put that all
# together to visualize a particular key-value pair!
#
# You might notice that this particular key (if you're using the pre-trained
# model weights) looks visually kind of like a nine, and lo and behold, when you
# go to the value vector that is getting written out, the highest activation is
# a 9!

visualize_ith_key_value(models[14].cpu(), 246)

# Go ahead and play around with other key value pairs and see if you can make
# sense of them.

# %%

# It's often useful to find which value vectors we have that tend to write
# strongly for certain kinds of digits.
#
# Here is one very rough stab at the problem that just looks for any value
# vector that has a value over a certain threshold for that digit. We'll quickly
# show a slightly less rough stab in just a moment.

def find_values_for_digit_over_threshold(model, digit, threshold=0.3):
  return torch.tensor([idx for idx in range(model.fc2.weight.shape[1]) if model.fc2.weight[digit, idx] > threshold])

#%%

find_values_for_digit_over_threshold(models[14], 0, threshold=0.4)

# Feel free to feed this into visualize_ith_key_value to see what that key_value pair looks like!

visualize_ith_key_value(models[14], 51440)

# %%

# Let's see a little bit more of how this key-value reframing of a vanilla
# neural net can help us understand things better.
#
# For example, we might hypothesize that the key which corresponds to a value vector
# that has a large positive value at 0 and small magnitude values for all other digits
# should look like a circle.
#
# Note that this is not obviously true! It might be the case that a model pieces
# together a zero exclusively by piecing together different arcs of a circle
# with no key actually being a full circle.
#
# But we can go ahead and test that right now. First we'll need to build a
# function that can find those key-value pairs which have values concentrated
# mostly on one digit and not as much on the others.
#
# This can be a bit finicky and hard to specify, so we've provided a
# rough-and-tumble version for you to use right here.

def find_values_for_sole_digit(model, digit, digit_threshold=0.16, other_digits_threshold=0.07):
  result = []
  for idx in range(model.fc2.weight.shape[1]):
    other_digits = torch.abs(model.fc2.weight[:, idx])
    other_digits[digit] = 0
    max_of_other_digits = torch.max(other_digits)
    if max_of_other_digits.item() > other_digits_threshold:
      continue
    elif model.fc2.weight[digit, idx] > digit_threshold:
      result.append(idx)
  return torch.tensor(result)

# %%

# TODO: find those key-value pairs which tend to write very strongly to the 
# digit 2, but very little for anything else using the above function
# raise NotImplementedError()

find_values_for_sole_digit(models[14], 2)

#Feel free to get a look at these!

# %%

# Let's now find those key-value pairs which tend to write strongly to the digit
# 0, but very little for everything else, and just analyze the first three of
# those key-value pairs. This will let us validate our hypothesis of whether we
# have keys that are looking for circles, or just fragmentary arcs of circles.

digit_to_analyze = 0

indices_that_fire_mainly_on_select_digit = find_values_for_sole_digit(models[14], digit_to_analyze)
for idx in indices_that_fire_mainly_on_select_digit[:3]:
  visualize_ith_key_value(models[14].to("cpu"), idx)

# %%

# TODO: Look at the results and what they tell you. Talk with your partner or
# group about what you're seeing. Once you've done that, delete this
# NotImplementedError and move on.
# raise NotImplementedError()

# %%

# This function will give us the internal outputs of all the keys and values for
# a given image. In other words this will return the dot product of each key
# with the image (combined with the bias per key) and will also return the
# scaled value vector.
#
# If this is confusing to you, it may be helpful to go back to the slides and
# look a little bit more at the break-down of how exactly we calculate a neural
# net's output using the key-value paradigm.

def compute_kv_outputs_for_image(model, input_image):
  flattened_img = model.flatten(input_image)
  output_after_keys = model.fc1(flattened_img)
  output_after_relu = model.relu(output_after_keys)
  # We ultimately want to multiple all the components of each value vector by
  # the same value, so we need to do a repeat first and then we can do a
  # standard element-wise tensor multiplication
  #
  # But this is just the same as broadcasting, so we just use that instead
  output_after_values = model.fc2.weight * output_after_relu
  return output_after_keys, output_after_values

# %%

def top_indices_by_tail_sum(tensor: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Given a 1D tensor and a threshold, returns the indices of the largest values
    such that the sum of all smaller values (i.e. the “tail” after that point)
    is <= threshold.
    """
    assert tensor.dim() == 1, "Only works on 1D tensors"
    # Sort descending
    sorted_vals, sorted_idx = tensor.sort(descending=True)
    # Compute cumulative sum of the sorted values
    cumsum = sorted_vals.cumsum(dim=0)
    total = cumsum[-1]
    # tail_sums[i] = sum(sorted_vals[i+1:])
    tail_sums = total - cumsum
    # find the first position where tail_sums <= threshold
    mask = tail_sums <= threshold
    if not mask.any():
        # no cutoff—tail never drops below threshold, so return empty
        return torch.empty(0, dtype=torch.long)
    cutoff = mask.nonzero(as_tuple=False)[0].item()
    # keep everything up to and including cutoff
    return sorted_idx[:cutoff + 1]

# Example
x = torch.tensor([1, 4, 2, 3, 1], dtype=torch.float)
indices = top_indices_by_tail_sum(x, threshold=4)
print(f"{indices=}")  # tensor([1, 3])

#returns the most influential key-value pairs for an image
def list_top_kv_pair_idxs(model, input_image, excess_abs_weight=500):
  _, output_after_values = compute_kv_outputs_for_image(model, input_image)
  abs_values = einops.einsum(torch.abs(output_after_values), "digits num_of_values -> num_of_values")
  indices = top_indices_by_tail_sum(abs_values, excess_abs_weight)
  return indices

# %%

# Let's prove to ourselves that the key-value paradigm of calculating things is equal to the normal layer-by-layer interpretation
def sanity_check_kv_outputs(model, input_image):
  _, output_after_values = compute_kv_outputs_for_image(model, input_image)
  output_plus_bias = einops.einsum(output_after_values, "digits num_of_values -> digits") + model.fc2.bias
  print(f"{output_plus_bias.softmax(dim=-1)=}")
  print(f"{model(input_image)=}")

# You should see that the two print statements print the same values
sanity_check_kv_outputs(models[14], train_dataset[0][0].cpu())

# %%

# This will list the key-value pairs that write the value vectors with the largest magnitude.
list_top_kv_pair_idxs(models[14], train_dataset[0][0].cpu(), 7000)

# %%

visualize_image(train_dataset[0][0].cpu().squeeze())

# %%

# Just an example, and you can try different keys as well!
visualize_ith_key_value_on_image(models[14], 14219, train_dataset[0][0].cpu().squeeze())

# %%

#finds the most variable key-value pairs
def sort_by_value_variance(model, input_image):
  _, output_after_values = compute_kv_outputs_for_image(model, input_image)
  print(f"{torch.var(output_after_values, dim=-1, keepdim=True).shape=}")
  variances = torch.var(output_after_values, dim=0, keepdim=True)
  var_values, var_indices = torch.sort(variances, dim=-1, descending=True)
  print(f"{var_indices.shape=}")
  return var_indices

top_5_kv_pairs_by_value_variance = sort_by_value_variance(models[14], train_dataset[0][0].cpu())[:, :5]
print(top_5_kv_pairs_by_value_variance)

# %%

visualize_ith_key_value_on_image(models[14], 22650, train_dataset[0][0].cpu().squeeze())

# %%

#finds key-value pairs that react almost only to one digit
def find_values_with_mostly_zeroes(model):
  values = model.fc2.weight
  num_of_elems_close_to_0 = torch.abs(values) < 0.05
  print(f"{values.shape=}")
  print(f"{num_of_elems_close_to_0.shape=}")
  nine_elems_close_to_0 = torch.sum(num_of_elems_close_to_0, dim=0) == 9
  indices_with_one_non_zero_elem = torch.nonzero(nine_elems_close_to_0).squeeze()
  large_total_sums = torch.nonzero(torch.sum(values, dim=0) > 0.18).squeeze()
  print(f"{indices_with_one_non_zero_elem.shape=}")
  large_total_sum_and_nine_elems_close_to_0 = indices_with_one_non_zero_elem[torch.isin(indices_with_one_non_zero_elem, large_total_sums)]
  print(f"{large_total_sum_and_nine_elems_close_to_0=}")

find_values_with_mostly_zeroes(models[14])

# %%

# TODO: visualize the two key-value pairs, especially pair 905
# raise NotImplementedError()

visualize_ith_key_value(models[14].cpu(), 905)

# %%

# this is an image of a 2, the digit which pair 905 reacts strongly to 
visualize_image(train_dataset[5][0].cpu().squeeze())

# %%

list_top_kv_pair_idxs(models[14], train_dataset[5][0].cpu(), 5500)

# %%

# However, what do you see in the interaction between pair 905 and the image?
visualize_ith_key_value_on_image(models[14].cpu(), 905, train_dataset[5][0].cpu().squeeze())

# %%

visualize_image(train_dataset[3][0].cpu().squeeze())

# %%

# Let's now look at the smallest model
# and compare it to what we saw before
list_top_kv_pair_idxs(models[0].cpu(), train_dataset[3][0].cpu(), 10_000)

# %%

models[0].cpu()(train_dataset[3][0].cpu()).squeeze()

# %%

visualize_ith_key_value_on_image(models[0].cpu(), 7, train_dataset[3][0].cpu().squeeze())

# %%

# This finds the image that activates mostly strongly for a given key.

def sort_highest_activating_image_for_key(model, key_value_idx, input_images):
  key = model.fc1.weight[key_value_idx, :]
  print(f"{input_images.shape=}")
  flattened_images = model.flatten(input_images)
  dot_products = einops.einsum(key, flattened_images, "key_dim, batch key_dim -> batch")
  _, indices_by_dot_product = torch.sort(dot_products, descending=True)
  return indices_by_dot_product

train_images = torch.stack([img for img, _ in train_dataset])

result = sort_highest_activating_image_for_key(models[14].cpu(), 905, train_images.cpu())

print(f"{result=}")

visualize_image(train_images[result][5].cpu().squeeze())

# %%

visualize_ith_key_value_on_image(models[14].cpu(), 905, train_dataset[2019][0].cpu().squeeze())

# %%

import copy

def delete_by_index(x: torch.Tensor, indices, dim: int = 0):
    """
    Return a new tensor with the specified indices removed along `dim`.

    Args
    ----
    x (torch.Tensor): input tensor
    indices (Sequence[int] | torch.Tensor): positions to delete
    dim (int): dimension along which to delete (default 0)

    Example
    -------
    >>> t = torch.tensor([[10, 11],
    ...                   [20, 21],
    ...                   [30, 31],
    ...                   [40, 41]])
    >>> delete_by_index(t, [1, 3])
    tensor([[10, 11],
            [30, 31]])
    """
    # Ensure we have a 1-D LongTensor of unique, sorted indices on the same device
    idx = torch.as_tensor(indices, dtype=torch.long, device=x.device).unique().sort().values

    # Build a boolean mask that is False at the indices we want to drop
    mask_shape = [1] * x.dim()
    mask_shape[dim] = x.size(dim)
    mask = torch.ones(mask_shape, dtype=torch.bool, device=x.device).squeeze()
    mask[idx] = False

    return x[mask] if dim == 0 else x.transpose(0, dim)[mask].transpose(0, dim)

#removes a certain key from the model
def knock_out_ith_key(model: SimpleNN, key_value_idx: torch.Tensor) -> SimpleNN:
  with torch.no_grad():
    new_model = copy.deepcopy(model)
    new_model.fc1 = torch.nn.Linear(model.fc1.in_features, model.fc1.out_features - key_value_idx.shape[0])
    new_model.fc2 = torch.nn.Linear(model.fc2.in_features - key_value_idx.shape[0], model.fc2.out_features)
    new_model.fc1.weight = torch.nn.Parameter(delete_by_index(model.fc1.weight, key_value_idx))
    new_model.fc1.bias = torch.nn.Parameter(delete_by_index(model.fc1.bias, key_value_idx))
    new_model.fc2.weight = torch.nn.Parameter(delete_by_index(model.fc2.weight, key_value_idx, dim=1))
    return new_model

# %%

# Find all those key-value pairs which activate a lot for zero
all_values_that_activate_significantly_for_zero = find_values_for_digit_over_threshold(models[14], 0, threshold=0.1)

# %%

# Let's see if we can just selectively knock those out!
model_with_0_knocked_out = knock_out_ith_key(models[14], all_values_that_activate_significantly_for_zero)

# %%

# And now we see that the model is basically entirely incapable of recognizing 0, but the rest of its capabilities are left intact!
accuracy_by_digit(model_with_0_knocked_out.to(DEVICE), test_loader)

# %%
