# Mechanistic Interpretability (Vanilla Neural Nets)

+ Assume you know what a neural net is
+ What is mechanistic interpretability?
    - Interpretability: trying to find a way to understand something
    - Mechanistic: finding the specific, concrete mechanisms used
        * Ideally you could drill down to the per-neuron level
+ We want to delve into the black box

# Common themes of mechanistic interpretability

+ Finding mathematically equivalent or near-equivalent restructurings of an
  architecture
+ Proving causality
+ Here, we are analyzing exactly how our model processes images!

# Our object of study

+ Vanilla neural net (single hidden layer)
+ Much simpler than modern LLMs
+ But still complex enough to demonstrate the fundamental themes of mechanistic interpretability

# What you'll be able to do at the end of the day

+ Get a new look at the traditional vanilla neural net architecture
+ Thoroughly and mechanistically explain why the NN gets certain digits wrong (and right)

# Standard way of thinking about a single hidden layer NN

+ There are two (three if you count inputs as neurons) layers of neurons
+ We pass iteratively from one layer to the next
+ Therefore to understand why 

# Diagram of standard neural net

<img src="./neural-net.svg"/>

# Standard layer by layer decomposition

<img src="./neural-net-layer-decomposition.svg" width="500px"/>

# Alternative decomposition: key-value

<img src="./neural-net-key-value-decomposition.svg" width="700px"/>

# Calculate as "key-value" decomposition

<img src="./neural-net-key-value-calculation.svg" width="500px"/>

# Concrete Example

<img src="./neural-net-example-0.svg" width="700px"/>

# Concrete Example

<img src="./neural-net-example-1.svg" width="700px"/>

# Concrete Example

<img src="./neural-net-example-2.svg" width="700px"/>

# Concrete Example

<img src="./neural-net-example-3.svg" width="700px"/>

# Concrete Example

<img src="./neural-net-example-4.svg" width="700px"/>

# Exercise

<img src="./exercise-question.jpeg" width="700px"/>

# Calculation With Traditional Layer-by-Layer Interpretation

<img src="./traditional-solution.jpeg" width="700px"/>

# Calculation With Key Value Decomposition

<img src="./key-value-solution.jpeg" width="700px"/>
