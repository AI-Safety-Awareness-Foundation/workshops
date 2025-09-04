# %% [markdown]
"""
This notebook is meant to be an interactive exploration of how we can use a
key-value perspective on a single hidden-layer neural net to get much more
interpretable results about how a basic neural net is able to do image
recognition of hand-written digits from the MNIST dataset.

This is meant to be used alongside a set of slides found in the associated
GitHub repository.

By the end of this set of exercises you will have acccomplished the following:

1. Analyzed why a vanilla neural net was able to correctly identify certain
handwritten digits
2. Analyzed why a vanilla neural net got some handwritten digits wrong
3. Used this understanding to specifically knock out the neural nets ability to
recognize one specific digit

Taken together, this means you will have gained a level of insight into a
vanilla neural net which makes the net significantly less "black box-y" than is
usually claimed. Along the way you'll get a bit of a taste of how mechanistic
interpretability works, albeit for a much more simplified model than what we
encounter in cutting-edge AI.
"""

# %% [markdown]
"""
__If you are running this in Google Colab make sure to uncomment the following
lines of code that are prefixed by exclamation marks.__
"""

# %%

# Not provided by default in Google Colab
#!pip install jaxtyping

# The rest downloads some binary files we'll need
#!pip install gdown

# Download all the pretrained models

#!gdown --id 1F2Z8ziHPaXd_GT_fYe974ySiQVhW0yz0
#!gdown --id 1gGE9MtYvQwCevY9qx-3BEYyoIhDOOBBh
#!gdown --id 1mLEVHleRHiGLHKvu06LE0Oq2Mm6b2lCg
#!gdown --id 1KTZ9m4qmEmUIr-FZMSeVUkd4H8LSB_4b
#!gdown --id 1-Q16KxTkfg5hktOAE-OPgWglgGaaaHFd
#!gdown --id 1RFwpFgpPIOONABfXiHGsMtJdUF1gd10S
#!gdown --id 15BUINas3RQcUVml3QQvoOTVkZjd7IPFS
#!gdown --id 1ASEwxJHndnjFiu2G9tdmn3B5OJkvRQb7
#!gdown --id 1AggdX5mQ9o1QAy9P8_ZpckC8qRDTwPm1
#!gdown --id 1nMjFaMMtijoLidaJSMQoYGnQld5TKp7Q
#!gdown --id 19H7Y50yQsUBj9dWzHWOU8VJJ6zU4cQhA
#!gdown --id 1A5O3OatMM0Lj2m1655Nx1NSwoervV0mS
#!gdown --id 1rKFm7BY8eqQitHYDTrKHzWoFDE_e7fp0
#!gdown --id 1vNM6pD5gc4oOzumGtURp6ASw7nqtwinz
#!gdown --id 1_1GFcsjIuWRUcgPU9rMEY5g66K7eX93M

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

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    assert len(x.shape) >= 3, f"We expected a tensor with at least 3 dimensions, not {len(x.shape)} dimensions (the overall shape was {x.shape}). The reason we expect 3 dimensions (and not say just 2 dimensions for a single image with two dimensions), is that the neural net expects entire batches of input, e.g. if you have three 28x28 images, you should stack them together to make a 3x28x28 tensor. If you have only one 28x28 image, you should use unsqueeze to make it a 1x28x28 batch of images"
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
    plt.text(x, 0, f'{value[0, x].item():.3f}', ha='center', va='center', color='black', fontsize=8)
  plt.axis('off')
  plt.title(f'Value {i}')
  plt.show()

#visualizes the global value bias for each digit, or the baseline before any interactions
def visualize_value_bias(model):
  value = model.fc2.bias.unsqueeze(0)
  norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
  plt.imshow(value.detach().numpy(), cmap='seismic', norm=norm)
  for x in range(value.shape[1]):
    plt.text(x, 0, f'{value[0, x].item():.3f}', ha='center', va='center', color='black', fontsize=8)
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
example_nn = SimpleNN(hidden_dim=2, input_dim=4, output_dim=2, has_bias=False, use_softmax=False)
preset_fc1 = nn.Parameter(
  torch.Tensor(
    [
      [1, 0, 0, 0],
      [0, 0, 0, 1],
    ]
  )
)
example_nn.fc1.weight = preset_fc1

preset_fc2 = nn.Parameter(torch.Tensor(
   [
      [1, 0.5],
      [0, 0.5],
   ]
))
example_nn.fc2.weight = preset_fc2

# %% [markdown]
"""
Note that our neural net expects input images to be in batches. So for example,
you can't pass a single n x m image to the neural net, but must always turn it
into a b x n x m tensor.
"""

# %%

# For example, if you try to directly call our neural net on a single image, you
# will get an error.
try:
  example_nn(input_0)
except AssertionError as e:
  print(e)

# %%

# Instead we can use unsqueeze to turn this image into a one image batch.
example_nn(input_0.unsqueeze(dim=0))

# %%

# We can also pass all three images at the same time.
three_images_stacked = torch.stack([input_0, input_1, input_2])

example_nn(three_images_stacked)

# %% [markdown]
"""
Now let's use `visualize_ith_key_value` to visualize the first and second key
value pairs of our small neural net.
"""

# %%

# First key-value pair
visualize_ith_key_value(
  model=example_nn, 
  i=0, 
  key_x_size=2, 
  key_y_size=2
)

# %%

# Second key-value pair
visualize_ith_key_value(
  model=example_nn, 
  i=1, 
  key_x_size=2, 
  key_y_size=2
)

# %% [markdown]
"""
*Exercise*: Verify that the results of the above two visualizations match the
visualization of the two key-value pairs from the slides. Also verify that the
results of passing `three_images_stacked` to the neural net match what you would
expect given the key-value interpretation as laid out in the slides.
"""

# %% [markdown]
"""
You should run the following code block, but you do not need to actually read
it. It's stuff that's relevant for actually training our neural nets, which we
will not be doing today, as for time purposes we will provide you with
pretrained models.
"""

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

# %% [markdown]
"""
This next code block is worth reading a little. The main thing we should realize
here is that we are using the MNIST dataset of handwritten digits, and that we
have 14 different pretrained models of varying sizes (from hidden layer sizes of
8 to 131072) we can use.

As part of the bonus exercises, you can use different sized models. We will
mainly be using the model with hidden layer size 65536, as this strikes a nice
balance between a sufficiently complicated neural net and one that is unlikely
to cause an out of memory error on the free version of Google Colab.
"""
# %%
# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10000)

hidden_dims = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
models = [SimpleNN(hidden_dim) for hidden_dim in hidden_dims]

MODEL_IDX_WE_ARE_USING=13

# %% [markdown]
"""
Now that we have the MNIST dataset, we can examine some pictures from it. As a
reminder, our neural net will **recognize the handwritten digits from this
dataset and assign them a label of 0-9**.
"""

# %%
# To understand the indexing of a dataset, the first index is the image index
# and the second index is whether it is the image data itself or the label.
#
# For example let's assume the 12th image in our training dataset is an image of
# a 7. `train_dataset[12][0]` selects the 12th image's image data (which will be
# some 1x28x28 tensor). `train_dataset[12][1]` selects the 12th image's label
# data (which will be a vector 10 elements long with a 1 at the 8th index (7 + 1
# for zero indexing) and a 0 everywhere else).
image_in_training_set = train_dataset[0][0].cpu()

# We need to squeeze because MNIST by default has images with 4 dimensions:
# batch, color channel, height, and width. The color channel is always 1 since
# all the images are grayscale, so we can simply `squeeze` that dimension away.
visualize_image(image_in_training_set.squeeze())

# The same kind of images are used in the test dataset, we just don't
image_in_test_dataset = test_dataset[1][0].cpu()
visualize_image(image_in_test_dataset.squeeze())

# %% [markdown]
"""
Once more, you should run the following code block, but you do not need to actually read
it. This code block can be used to either execute the training loop of the model
or pre-load its weights. We're interested mainly in the latter for today, but if
you wanted to train these models from scratch, the code is provided for you.

To reiterate, even though you don't need to read this code block, it is vital to
run it! Otherwise you won't have pre-trained models, but rather models set to
random parameters.

Feel free to collapse and hide this code block after running if it's too distracting.
"""

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

if TRAIN_FROM_SCRATCH:
  for dim, model in zip(hidden_dims, models):
    # Save on CPU because this makes it easier to load for more devices
    model = model.to("cpu")
    torch.save(model.state_dict(), f"mnist_model_hidden_layer_{dim}")

# Go ahead and run this just to make sure we're on the correct device

for model in models:
  model = model.to(DEVICE)

# %% [markdown]
"""
As a reminder we're mainly only the `MODEL_IDX_WE_ARE_USING`-th model for most
of our exercises today.

Let's measure how accurately this model classifies digits. Accuracy will be
measured as a 10 element list, where the $i-1$-ith member of the list (since the
list is zero indexed) corresponds to the model's accuracy at correctly
identifying images of the $i$-th digit.

For example, `[0.9, 0.8, 0.7, 0.6, 0.5, 0.55, 0.65, 0.75, 0.85, 0.95]` means
that the model correctly identifies images of the digit 3 as 3 60% of the time
(since the fourth element of the list is 0.6).

You should notice that most of the accuracy scores are in the high nineties for
our model.

By the end of these exercises, you will see how to selectively drive down the
model's ability to recognize one digit down to near-zero accuracy with minimal
interference for other digits.
"""

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

# Test it out for our chosen model
accuracy_by_digit(models[MODEL_IDX_WE_ARE_USING], test_loader)

# %% [markdown]
"""
We can compare this accuracy against our smallest model, which is substantially
less accurate.
"""

# %%

# Test it out for our 8 hidden units model
accuracy_by_digit(models[0], test_loader)

# %% [markdown]
"""
We can now begin looking at much larger vanilla neural nets than what we were
looking at before. Using our 65k hidden unit model, let's look at one
interesting key-value pair, the 463rd one.

*Exercise*: For the 463rd key-value pair, given the representation of the key
below, what do you think the value vector looks like? In particular, which index
of the value vector do you think has the highest value? Remember that red in the
key visualization indicates a high positive value and blue indicates a high
negative value, while white indicates a zero value.

<details>
<summary>Solution</summary>
You should see that the image looks roughly like a circle, which indicates that
this is a key that recognizes 0s. This in turn means that the value vector
should likely write out most highly at the 0th position, and pretty low
elsewhere.
</details>
"""

# %%

INTERESTING_KEY_VALUE_PAIR = 463

visualize_ith_key(models[MODEL_IDX_WE_ARE_USING].cpu(), INTERESTING_KEY_VALUE_PAIR)

# %% [markdown]
"""
You can verify the solution to the exercise by printing the actual value vector.
"""

# %%
visualize_ith_value(models[MODEL_IDX_WE_ARE_USING].cpu(), INTERESTING_KEY_VALUE_PAIR)

# %% [markdown]
"""
You can also visualize both the key and the value at the same time (which is
what we did for the simpler neural net on 2x2 images earlier).
"""

# %%
visualize_ith_key_value(models[MODEL_IDX_WE_ARE_USING].cpu(), INTERESTING_KEY_VALUE_PAIR)

# %% [markdown]
"""
What is a more general way we could find interesting key-value pairs to look at?
Well we could look for those key-value pairs which selectively recognize one
particular kind of digit and not others.

*Exercise*: What kind of key and value vectors would be expect to see for
*key-value pairs which selectively recognize one digit and not others? Would we
*pay more attention to the key vector or the value vector?

<details>
<summary>Solution</summary>
For this it's generally more useful to look at the value vector, since this
provides the answer as to what the key-value pair "thinks" a given image is. In
particular, we would expect the value vector to have a high value for one index
and a low value at all other indices.
</details>
"""

# %% [markdown]
"""
The following function is one very simple operationalization of the previous
exercise.
"""

# %%

# This is a very rough-and-tumble function. We pass in the digit we're
# interested in and then just look at how high of a value the value vector has
# at the corresponding index and then how low the absolute values of all the other
# values are. If the value corresponding to the digit is high (i.e. over
# `digit_threshold`) and the values corresponding to the other digits are low
# (i.e. less than `other_digits_threshold`), then we keep that result.

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

# %% [markdown]
"""
Let's use `find_values_for_sole_digit` to find interesting key-value pairs that
very selectively recognize handwritten digits of 1s.
"""

# %%
find_values_for_sole_digit(models[MODEL_IDX_WE_ARE_USING], 1)

# %%
# You should find from the previous block that 7568 is one of the key-value
# pairs that very selectively recognizes 1s. Let's visualize that.

visualize_ith_key_value(models[MODEL_IDX_WE_ARE_USING].cpu(), 7568)

# %% [markdown]
"""
*Exercise*: Use the next code block to call `find_values_for_sole_digit` to find
interesting key-value pairs that very selectively recognize handwritten digits
of 0s. Do the keys actually look like zeros?
"""

# %%
# TODO: Fill in this code block with the proper call to
# find_values_for_sole_digit and then visualize each of the key-value pairs from
# the call!

# raise NotImplementedError("")

#BEGIN SOLUTION
digit_to_analyze = 0

indices_that_fire_mainly_on_select_digit = find_values_for_sole_digit(models[MODEL_IDX_WE_ARE_USING], digit_to_analyze)
for idx in indices_that_fire_mainly_on_select_digit[:3]:
  visualize_ith_key_value(models[MODEL_IDX_WE_ARE_USING].to("cpu"), idx)
#END SOLUTION

# %% [markdown]
"""
We've been able to identify interesting key-value pairs, but we have no
guarantee that those key-value pairs are actually the most important ones that
the model uses to do image recognition.

This is an important problem within mechanistic interpretability: how do we
*attribute* certain kinds of behaviors to certain structures within the neural
net?

In a certain sense, starting with interesting structures is a bit backwards,
since have no idea if those structures actually are the main way that the model
recognizes handwritten digits or if they're just mostly unused anomalies.

We won't have time today to thoroughly explore attribution, other than to
mention that it's tricky! Even for a vanilla neural net it can be non-obvious
how to actually robustly perform attribution.

For now, we'll just look at some individual images and try to do some very rough
attribution on the fly for each of them. Let's begin with an image from the
model's training set, an image of a 0.
"""

# %%
# We explored this earlier, but we'll copy this again from before.
#
# To understand the indexing of a dataset, the first index is the image index
# and the second index is whether it is the image data itself or the label.
#
# For example let's assume the 12th image in our training dataset is an image of
# a 7. `train_dataset[12][0]` selects the 12th image's image data (which will be
# some 1x28x28 tensor). `train_dataset[12][1]` selects the 12th image's label
# data (which will be a vector 10 elements long with a 1 at the 8th index (7 + 1
# for zero indexing) and a 0 everywhere else).
image_of_zero_in_training_set = train_dataset[1][0].cpu() # I just happen to know that the 1-th element is an image of a 0

# We need to squeeze because MNIST by default has images with 4 dimensions:
# batch, color channel, height, and width. The color channel is always 1 since
# all the images are grayscale, so we can simply `squeeze` that dimension away.
visualize_image(image_of_zero_in_training_set.squeeze())

# %% [markdown]
"""
*Exercise*: Visualize the image of the 10th picture in our training set
(including zero indexing, so you should be using 10 as index for the dataset).
"""

# %%
# TODO: First correctly index `train_dataset`, and then call `visualize_image`
# raise NotImplementedError("")

#BEGIN SOLUTION
visualize_image(train_dataset[10][0].cpu().squeeze())
#END SOLUTION

# %% [markdown]
"""
This next block of code attempts to perform some very rough attribution

It's a significant amount of code, but it's not necessary to read all of it to
understand what it's doing. Feel free to collapse this cell after running it.

It all culminates in `list_top_kv_pair_idxs`, which is a function that finds
those key-value pairs which "activate" the most on a given image. We measure
activation of a key-value pair by looking at the sum of the absolute values of
the final scaled value vector written by the key-value pair. 

A high number indicates that we have a key-value pair that potentially was very
instrumental in helping the neural net identify the digit. A value close to zero
indicates that that key-value pair had very little impact on the neural net's
evaluation of that image.

`list_top_kv_pair_idxs` gives back a 1-d tensor of those key-value pairs which
contributed the most according to this simple activation definition to the
neural net's evaluation of a particular image, sorted by activation amount.

`excess_abs_weight` is a parameter that controls how many key-value pairs we
ignore and do not return in the final result. So for example an
excess_abs_weight of 1000 means that we discard all the key-value pairs with the
lowest activation scores such that their total sum does not exceed 1000.

Higher values of `excess_abs_weight` will decrease the size of the tensor that
is returned because more key-value pairs will be pruned out. 

If you require very high levels (e.g. > 2000) of `excess_abs_weight` to prune out key-value
pairs, this generally indicates that the model is spreading most of its
evaluation across many different key-value pairs.

Low levels of `excess_abs_weight` which prune out most key-value pairs on the
other hand indicate that for that particular image, the model evaluates it
mainly just using a few key-value pairs.
"""

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
    return sorted_idx[:cutoff]

# Example
x = torch.tensor([1, 4, 2, 3, 1], dtype=torch.float)
indices = top_indices_by_tail_sum(x, threshold=4)
# print(f"{indices=}")  # tensor([1, 3])

# Let's prove to ourselves that the key-value paradigm of calculating things is equal to the normal layer-by-layer interpretation
def sanity_check_kv_outputs(model, input_image):
  _, output_after_values = compute_kv_outputs_for_image(model, input_image)
  output_plus_bias = einops.einsum(output_after_values, "digits num_of_values -> digits") + model.fc2.bias
  # Uncomment these if you want to actually see the sanity check
  # print(f"{output_plus_bias.softmax(dim=-1)=}")
  # print(f"{model(input_image)=}")

# You should see that the two print statements print the same values
sanity_check_kv_outputs(models[MODEL_IDX_WE_ARE_USING], train_dataset[0][0].cpu())

#returns the most influential key-value pairs for an image
def list_top_kv_pair_idxs(model, input_image, excess_abs_weight=500):
  _, output_after_values = compute_kv_outputs_for_image(model, input_image)
  abs_values = einops.einsum(torch.abs(output_after_values), "digits num_of_values -> num_of_values")
  indices = top_indices_by_tail_sum(abs_values, excess_abs_weight)
  return indices

# %% [markdown]
"""
Now let's find those key-value pairs that activate most strongly for the image of a zero we saw a while ago.
"""

# %%
# This will list the key-value pairs that write the value vectors with the largest magnitude.
top_key_value_pairs_for_img_of_zero = list_top_kv_pair_idxs(models[13], image_of_zero_in_training_set, 1800)
top_key_value_pairs_for_img_of_zero

# %%

# 42138 is the key-value pair with the highest activation for this particular image
first_highest_activation_kv = top_key_value_pairs_for_img_of_zero[0]
print(f"{first_highest_activation_kv=}")
visualize_ith_key_value_on_image(models[MODEL_IDX_WE_ARE_USING], first_highest_activation_kv, image_of_zero_in_training_set.squeeze())
visualize_image(image_of_zero_in_training_set.squeeze())

# %% [markdown]
"""
*Exercise*: Visualize the 2nd and 3rd most activating key-value pairs for
`image_of_zero_in_training_set`. Do you notice anything interesting about them?
In particular, do they seem to select for digits that aren't just 0? If so,
which ones and does it make sense that they are selecting for those as well?

<details>
<summary>Solution</summary>
You should notice that all these highly fire for 2s as well. This makes sense
because the part of the 2 that is not the bottom horizontal line has a lot of
overlap with 0.
</details>
"""

# %%
# TODO: Fill in code block with correct calls to visualize the 2nd and 3rd most
# activating key pairs for `image_of_zero_in_training_set`

# raise NotImplementedError()
#BEGIN SOLUTION
second_highest_activation_kv = top_key_value_pairs_for_img_of_zero[1]
print(f"{second_highest_activation_kv=}")
visualize_ith_key_value_on_image(models[MODEL_IDX_WE_ARE_USING], second_highest_activation_kv, image_of_zero_in_training_set.squeeze())
visualize_image(image_of_zero_in_training_set.squeeze())

third_highest_activation_kv = top_key_value_pairs_for_img_of_zero[2]
print(f"{third_highest_activation_kv=}")
visualize_ith_key_value_on_image(models[MODEL_IDX_WE_ARE_USING], third_highest_activation_kv, image_of_zero_in_training_set.squeeze())
visualize_image(image_of_zero_in_training_set.squeeze())
#END SOLUTION

# %% [markdown]
"""
`top_key_value_pairs_for_img_of_zero` gives us a basic way of doing attribution,
but we're still missing a crucial piece in how we perform attribution, namely
the ability to see how the network performs when we only use certain key-value
pairs (or equivalently when we omit certain ones).

This lets us get more confidence we understand how the network is recognizing
images by letting us see if the removal of what we think are relevant neurons
actually removes the behavior we are analyzing or whether their inclusion
maintains the behavior.

The next function is developed for this purpose. It returns both logits (raw
scores that are not softmax-ed) and the softmax-ed probabilities. The latter is
technically a bit harder to understand than the former, because it doesn't
compose as nicely. The logits of different key-value pairs are just summed
together by the neural net to get the final answer, but softmax is a non-linear
function that makes attribution a little murkier.

We include the softmax-ed results nonetheless because they are useful for
building intuition.
"""

# %%

#function to find the logits and probabilities if the model only uses a certain set of key-value pair indices
def calculate_output_only_with_certain_kv_indices(model, img, index_list):

  image = img.unsqueeze(0)
  image = model.flatten(img)
  index_list = torch.tensor(index_list)

  all_outputs = model.fc1(image)
  all = model.relu(all_outputs)

  zeroing = torch.zeros_like(all[0])
  zeroing[index_list] = 1.0

  all = all * zeroing

  logits = model.fc2(all)
  probabilities = model.softmax(logits)

  return logits, probabilities

# %% [markdown]
"""
With this function in hand, we can verify that in fact if we restrict the
network to just using the top key-value pairs we found previously, it does still
recognize the image as a zero.
"""

# %%
# You should notice that even with just the 13 key-value pairs in
# `top_key_value_pairs_for_img_of_zero`, our network is already able to quite
# confidently recognize this digit as a 0.
calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_zero_in_training_set,
  top_key_value_pairs_for_img_of_zero,
)

# %% [markdown]
"""
But you might notice a bit of an oddity here. The top 3 highest activating
key-value pairs for this image of a 0 actually also highly activate for 2. In
fact several of them write a value vector with values higher for a 2 than a 0!

What's going on here? Let's explore that a little. First, we can notice that if
we only include the top 5 highest activating key-value pairs, the neural net
thinks that the image is more of a 2 than a 0.
"""

# %%
calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_zero_in_training_set,
  top_key_value_pairs_for_img_of_zero[:5],
)

# %% [markdown]
"""
So that means somewhere between the top 5 highest activating key-value pairs and
the remaining 8 highest activating key-value pairs (because there are a total of
13 highest activatig key-value pairs we're using), the model "flipped" and
decided that the image was much of a 0 than a 2.

Let's visualize each of the remaining key-value pairs to see why that happened.
"""

# %%

for i in top_key_value_pairs_for_img_of_zero[5:]:
  visualize_ith_key_value_on_image(models[MODEL_IDX_WE_ARE_USING], i, image_of_zero_in_training_set.squeeze())

# %% [markdown]
"""
*Exercise*: Using these visualizations, and any others you see fit to use, can
you come up with an explanation for how the model decides that this is more of a
0 than a 2? Can you identify specific key-value pairs/neurons that are
responsible for this behavior?

<details>
<summary>Solution</summary>
There's some variation in possible answers here, but you might notice after some
exploration that really what's causing the neural net to "hesitate" between a 0
and a 2, is the presence of keys that activate heavily on the lower part of a 0,
which could correspond either to the bottom of a zero, or to the bottom
horizontal line of a 2.

If you look at some of the other highly-activating key-value pairs, you should
find that some of them heavily favor 0 over 2, because the keys activate a lot
less in that region (and don't have something that looks closer to a horizontal
line).

So one version of an explanation could go: the neural net ultimately tie-breaks
between 0 and 2 by using a lot of keys which do not have very "long horizontal"
streaks at the bottom of a 0 digit to select for the 0 over a 2.

Feel free to look in the solutions file for examples of some scratch code I
wrote to come to this conclusion.
</details>

"""

# %%
# This function might come in handy (but you don't have to use it). It shows you
# the specific key and value activation results for a certain key-value pair
# when applied to a single image.
#
# You can also just manually inspect each of the 13 key-value pairs yourself.
def calculate_kv_activation_for_specific_kv(model, img, kv_idx):
  keys, values = compute_kv_outputs_for_image(model, img)
  return keys[:, kv_idx], values[:, kv_idx]

# TODO: Scratch space for any code you want to write to come up with an explanation.

# The following is one example of some code you might write
key_value_indices_that_prefer_0_over_2 = []
for kv_pair_idx in top_key_value_pairs_for_img_of_zero[5:]:
  _, values = calculate_kv_activation_for_specific_kv(models[MODEL_IDX_WE_ARE_USING], image_of_zero_in_training_set, kv_pair_idx)
  if values[0] > values[2]:
    key_value_indices_that_prefer_0_over_2.append(kv_pair_idx)

# Let's visualize all of the key-value pairs that prefer 0 over 2.
for kv_pair_idx in key_value_indices_that_prefer_0_over_2:
  visualize_ith_key_value_on_image(models[MODEL_IDX_WE_ARE_USING], kv_pair_idx, image_of_zero_in_training_set.squeeze())

# %% [markdown]
"""
At this point we have some idea of how the model is able to recognize our chosen
image as a 0. It has keys that fire heavily on the curves of the 0, especially
the right-hand curve, and then uses some additional keys that don't have a
"bottom curve" component to disambiguate between 0 and 2.

Let's move on to an image from the training data set that is more difficult to
interpret.
"""

# %%
image_of_one_in_training_set = train_dataset[3][0].cpu()
visualize_image(image_of_one_in_training_set.squeeze())

# %% [markdown]
"""
Let's look again at the top key-value pairs. Right off the bat, you should
notice we have a *lot* more.

This already indicates that we're going to have a tougher time interpreting
what's going on. Instead of being able to confine our attention to a handful of
key-value pairs, we potentially have to do deal with way more.
"""

# %%
top_key_value_pairs_for_img_of_one = list_top_kv_pair_idxs(models[13], image_of_one_in_training_set, 1800)
top_key_value_pairs_for_img_of_one

# %% [markdown]
"""
Let's do a quick sanity check to make sure that when we confine the neural net
to just using these top key-value pairs, it still recognizes this image as a 1.
"""

# %%
calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_one_in_training_set,
  top_key_value_pairs_for_img_of_one,
)

# %% [markdown]
"""
Now let's look at one of the particular key-value pairs. This one is really
strange. It's the 6th highest (remember zero-indexing, so we use 5 to index into
the list) activating key-value pair, but hardly thinks the image is a 1 at all!
Instead it most strongly thinks it's an 8, and also might be a 2 or a 6.
"""

# %%
visualize_ith_key_value_on_image(
  models[MODEL_IDX_WE_ARE_USING], 
  top_key_value_pairs_for_img_of_one[5], 
  image_of_one_in_training_set.squeeze()
)

# %% [markdown]
"""
Indeed if we look at the top 10 highest activating key-value pairs, taking
together, they think that the image is probably a 2 or an 8.
"""

# %%
calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_one_in_training_set,
  top_key_value_pairs_for_img_of_one[:10],
)

# %% [markdown]
"""
Only when you expand out to around the top 44 or so, then finally, begrudgingly,
the model thinks that the image is likely a 1.
"""

# %%
calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_one_in_training_set,
  top_key_value_pairs_for_img_of_one[:44],
)

# %% [markdown]
"""
*Exercise*: Attempt to explain how the model is able to conclude that this image
is in fact a 1. Why do you think that how the neural net is able to conclude
that the image is a 1 is far more messy than for a 0?

<details>
<summary>Solution</summary>
The explanation here is a lot more tentative than for our previous image. You
should retain some skepticism about this and if you have time, feel free to more
thoroughly test this explanation!

It looks like basically there's a lot of interference from other digits because
a diagonal line (and most 1s in the dataset are written as a diagonal line
rather than e.g. a vertical line) will intersect with many other keys (e.g. a 2
has a digonal line in it, an 8 written in a slanted fashion basically has a
diagonal line through it, and a 7 also has a diagonal in it).

So the neural net is forced to have a much more patchwork set of keys that all
activate on various different parts of a 1 and cancel out on various other parts
of other digits to piece them together.
</details>
"""

# %%
# TODO: Scratch space for any code you want to write to come up with an explanation.
# raise NotImplementedError()

#BEGIN SOLUTION
# Let's see which key-value pairs prefer 1.
key_value_indices_that_prefer_1 = []
for kv_pair_idx in top_key_value_pairs_for_img_of_one:
  _, values = calculate_kv_activation_for_specific_kv(models[MODEL_IDX_WE_ARE_USING], image_of_one_in_training_set, kv_pair_idx)
  if values.argmax() == 1:
    key_value_indices_that_prefer_1.append(kv_pair_idx)
print(f"{key_value_indices_that_prefer_1=}")
for idx in key_value_indices_that_prefer_1[:5]:
  visualize_ith_key_value_on_image(
    models[MODEL_IDX_WE_ARE_USING], 
    idx, 
    image_of_one_in_training_set.squeeze()
  )

# To get an idea of the interference we're getting, look at the those kv-pairs
# which think that the image is more of a 7 than a 1
key_value_indices_that_prefer_7_over_1 = []
for kv_pair_idx in top_key_value_pairs_for_img_of_one:
  _, values = calculate_kv_activation_for_specific_kv(models[MODEL_IDX_WE_ARE_USING], image_of_one_in_training_set, kv_pair_idx)
  if values[7] > values[1] and values[7] > 0.1:
    key_value_indices_that_prefer_7_over_1.append(kv_pair_idx)

# There's a lot of key-value pairs that prefer 7 over 1 here!
print(f"{key_value_indices_that_prefer_7_over_1=}")

for idx in key_value_indices_that_prefer_7_over_1[:5]:
  visualize_ith_key_value_on_image(
    models[MODEL_IDX_WE_ARE_USING], 
    idx, 
    image_of_one_in_training_set.squeeze()
  )
#END SOLUTION

# %% [markdown]
"""
This illustrates how even for our very simple neural net, mechanistic
interpretability can be quite difficult!

Let's see if we can quantify a little bit this difficulty. If we plot all the
positive key activations (i.e. the dot product of the key with the image + the
bias) and sort them by key activation amount, we can see that for the image of a
0, we have a pretty sharp curve. That is the majority of key activations are
squeezed into a relatively small number of keys.

If we compare that to the image of a 1, we see a much flatter curve, which
suggests that the key activations are much more "spread out" and we have to
understand a lot more keys to get a handle on why the image is recognized as a 1
by the model.
"""

# %%
#returns all positive key activations (dot products + bias) for each key for an image
def calculate_key_activation(model, img):
    viz_img = img
    visualize_image(viz_img)
    dot_products_with_bias = []

    with torch.no_grad():
        for i in range(model.fc1.weight.shape[0]):
            key = model.fc1.weight[i].reshape(28, 28)
            key_bias = model.fc1.bias[i]
            element_wise_multi = key * img
            dot = torch.sum(element_wise_multi)
            dot_with_bias = dot + key_bias

            if(dot > 0):
                dot_products_with_bias.append(dot_with_bias.item())

    return dot_products_with_bias

#plots the key activations, ensuring the x and y axis has the same scale each time
def plot_key_activations(dot_products_with_bias):
    plt.figure(figsize=(12, 8))

    if isinstance(dot_products_with_bias, torch.Tensor):
        dot_products_with_bias = dot_products_with_bias.numpy()

    sort_indices = np.argsort(dot_products_with_bias)
    sorted_dot_products_with_bias = np.array(dot_products_with_bias)[sort_indices]

    x_indices = range(len(sorted_dot_products_with_bias))

    plt.plot(x_indices, sorted_dot_products_with_bias, 'r-', label='Dot Product + Key Bias', linewidth=1.5, alpha=0.7)

    plt.xlabel('Keys (sorted by key activation, ascending)')
    plt.ylabel('Key Activation')
    plt.title('Distribution of Key Activations')
    plt.legend()
    plt.ylim(0, 10)
    plt.xlim(0, 5000)
    plt.grid(True, alpha=0.3)
    plt.show()

key_activations_for_image_of_zero = calculate_key_activation(models[MODEL_IDX_WE_ARE_USING].cpu(), image_of_zero_in_training_set.squeeze())
plot_key_activations(key_activations_for_image_of_zero)

key_activations_for_image_of_one = calculate_key_activation(models[MODEL_IDX_WE_ARE_USING].cpu(), image_of_one_in_training_set.squeeze())
plot_key_activations(key_activations_for_image_of_one)

# %% [markdown]
"""
Now let's go and analyze an image that the model gets wrong. This time we'll use
the test dataset to find such an image.

Here's an image of a 4 that the model really really thinks is a 2.
"""

# %%
image_of_four_in_test_set = test_dataset[247][0].cpu()
visualize_image(image_of_four_in_test_set.squeeze())
models[MODEL_IDX_WE_ARE_USING].cpu()(image_of_four_in_test_set)

# %% [markdown]
"""
Like before, we can use `list_top_kv_pair_idxs` to find a set of most signficant
key-value pairs
"""
# %%
top_key_value_pairs_for_img_of_four = list_top_kv_pair_idxs(models[MODEL_IDX_WE_ARE_USING].cpu(), image_of_four_in_test_set, 500)
print(f"{top_key_value_pairs_for_img_of_four=}")

# %% [markdown]
"""
Let's do a quick sanity check to make sure that restricting these top key value
pairs to an image of a four still has the model thinking that it's a 2.
"""

# %%
calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_four_in_test_set,
  top_key_value_pairs_for_img_of_four,
)

# %% [markdown]
"""
This image has a pretty interesting thing where if we look at the initial group
of top activating key-value pairs, the model initially thinks that the image is
in fact a 6.
"""

# %%
calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_four_in_test_set,
  top_key_value_pairs_for_img_of_four[:15],
)

# %% [markdown]
"""
*Exercise*: Why does the model think that the image is a 6 with only the first 15
top activating key-value pairs used? What is the general shape of the
keys that activate for this and can you make sense of this?

<details>
<summary>Solution</summary>
The model seems to detect sixes by mainly looking at blobs in the center of the
image with a little bit of some stuff above the blob. You can look e.g. at
`top_key_value_pairs_for_img_of_four[14]` or other examples in the solutions
file for an example of this.
</details>
"""

# %%
# TODO: Scratch space for any code you want to write to come up with an explanation.
# raise NotImplementedError()

#BEGIN SOLUTION
# These both seem to code heavily for sixes.
visualize_ith_key_value_on_image(
  models[MODEL_IDX_WE_ARE_USING], 
  top_key_value_pairs_for_img_of_four[14], 
  image_of_four_in_test_set.squeeze(),
)
visualize_ith_key_value_on_image(
  models[MODEL_IDX_WE_ARE_USING], 
  top_key_value_pairs_for_img_of_four[13], 
  image_of_four_in_test_set.squeeze(),
)
#END SOLUTION

# %% [markdown]
"""
In fact, all the way up to the first 500 most highly activating key-value pairs,
the model still thinks that the image is likely a 6.

Somewhere between about the first 500 and first 700 most highly activating
key-value pairs, the model goes from a 6 to a 2.
"""

# %%
# Notice that if go up to the first 600, we start getting more confidence that
# the image is a 2.
output_of_first_600_kv_pairs = calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_four_in_test_set,
  top_key_value_pairs_for_img_of_four[:600],
)
print(f"{output_of_first_600_kv_pairs=}")
# And that if you go to the first 700 then it is very confident that the image
# is a 2
output_of_first_700_kv_pairs = calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_four_in_test_set,
  top_key_value_pairs_for_img_of_four[:700],
)
print(f"{output_of_first_700_kv_pairs=}")

# %% [markdown]
"""
*Exercise*: So why does the model go from thinking that the image is a 6 to
thinking that the image is a 2?

<details>
<summary>Solution</summary>
The explanation here again is a lot more tentative than for our previous image. You
should retain some skepticism about this and if you have time, feel free to more
thoroughly test this explanation!

Basically, a lot of the key value pairs that strongly make the model think that
an image is a 2 do so with a component that looks for a strong horizontal stripe
(the bottom horizontal line of a 2). This just happens to line up very well with
the lower-most blob of our image of a 4, which looks kind of like a single
horizontal line.
</details>
"""

# %%

# It might be useful to start here with a list of all the kv pairs that activate
# strongly for 2 on this particular image
key_values_indices_that_code_strongly_for_2 = []

for kv_pair_idx in top_key_value_pairs_for_img_of_four[:600]:
  _, values = calculate_kv_activation_for_specific_kv(models[MODEL_IDX_WE_ARE_USING], image_of_four_in_test_set, kv_pair_idx)
  if values[2] > 0.5:
    key_values_indices_that_code_strongly_for_2.append(kv_pair_idx)

print(f"{key_values_indices_that_code_strongly_for_2=}")

# Unsurprisingly just using these values will make the model highly think that
# the image is a 2.
result_of_just_using_kv_indices_coding_strongly_for_2 = calculate_output_only_with_certain_kv_indices(
  models[MODEL_IDX_WE_ARE_USING], 
  image_of_four_in_test_set,
  key_values_indices_that_code_strongly_for_2,
)
print(f"{result_of_just_using_kv_indices_coding_strongly_for_2=}")

# TODO: Scratch space for any code you want to write to come up with an explanation.
# raise NotImplementedError()

#BEGIN SOLUTION
for kv_idx in key_values_indices_that_code_strongly_for_2[:10]:
  visualize_ith_key_value_on_image(
    models[MODEL_IDX_WE_ARE_USING], 
    kv_idx, 
    image_of_four_in_test_set.squeeze(),
  )
#END SOLUTION

# %% [markdown]
"""
Our final main objective, before our bonus exercises, is to use the knowledge
that we've gained here to selectively knock out the model's ability to recognize
a single digit, without needing to resort to gradient descent to retrain the
model.

If we can do that, we've demonstrated a certain level of surgical insight into
the model that goes beyond the standard "black box" thinking.

Make sure you understand what's going on with `knock_out_ith_key` and
`find_values_for_digit_over_threshold`.

We'll be using that to knock out the model's ability to recognize 0s.
"""

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

def find_values_for_digit_over_threshold(model, digit, threshold=0.3):
  return torch.tensor([idx for idx in range(model.fc2.weight.shape[1]) if model.fc2.weight[digit, idx] > threshold])

# %%

# Find all those key-value pairs which activate a lot for zero
all_values_that_activate_significantly_for_zero = find_values_for_digit_over_threshold(models[MODEL_IDX_WE_ARE_USING], 0, threshold=0.1)

# %%

# Let's see if we can just selectively knock those out!
model_with_0_knocked_out = knock_out_ith_key(models[MODEL_IDX_WE_ARE_USING], all_values_that_activate_significantly_for_zero)

# %%

# And now we see that the model is basically entirely incapable of recognizing 0, but the rest of its capabilities are left intact!
accuracy_by_digit(model_with_0_knocked_out.to(DEVICE), test_loader)

# %% [markdown]
"""
*Exercise*: Can you do a similar thing for the digit 9, where we knock out the model's ability to recognize 9s?
"""

# %%
# TODO: Scratchpad for exercise
# raise NotImplementedError()

#BEGIN SOLUTION
# Find all those key-value pairs which activate a lot for zero
all_values_that_activate_significantly_for_nine = find_values_for_digit_over_threshold(models[MODEL_IDX_WE_ARE_USING], 9, threshold=0.05)

# Let's see if we can just selectively knock those out!
model_with_9_knocked_out = knock_out_ith_key(models[MODEL_IDX_WE_ARE_USING], all_values_that_activate_significantly_for_nine)

# And now we see that the model is basically entirely incapable of recognizing 0, but the rest of its capabilities are left intact!
accuracy_by_digit(model_with_9_knocked_out.to(DEVICE), test_loader)
#END SOLUTION

# %% [markdown]
"""
*Bonus Exercise*: Can you put everything together that we've learned so far in
this workshop to craft an image that you think the neural net will get wrong?
Even better can you predict what label the net will assign to the image
instead?

Feel free to do whatever analysis you want on the net, the only constraint is
that you're not allowed to pass the image through the net itself (since that
would be giving away the answer!)

If you can do this without running the image through the net beforehand, this
will demonstrate that you've pretty deeply understood the net!
"""

# %%
# TODO: Scratchpad for exercise
# raise NotImplementedError()

#BEGIN SOLUTION
# Here's our example, where the model gets a 6 very wrong because it expects 6s
# to have big blobs in the middle of the image.
tester = (torch.tensor(
[[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]).to(torch.float))

visualize_image(tester)

models[13](tester.unsqueeze(0))
#END SOLUTION
# %% [markdown]
"""
*Bonus Exercise*: Visualize some of the key value pairs of the smallest model, both independently and on an image
Are the visualizations explainable? Could you infer what some of the key-value pairs are doing?
"""
