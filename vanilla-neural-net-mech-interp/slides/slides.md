# Mechanistic Interpretability (Vanilla Neural Nets)

+ Assuming you know what a neural net is
+ What is mechanistic interpretability?
    - Interpretability: trying to find a way
    - Mechanistic: finding the specific, concrete mechanisms used
        * Ideally you could drill down to the per-neuron level

# Common themes of mechanistic interpretability

+ Finding mathematically equivalent or near-equivalent restructuring of an
  architecture
+ Proving causality
+ Analyzing exactly how our model processes images!

# Our object of study

+ Vanilla neural net (single hidden layer)
+ Much simpler than modern LLMs
+ But still complex enough to demonstrate the fundamental themes of mech interp

# What you'll be able to do at the end of the day

+ Get a new look at the traditional vanilla neural net architecture
+ Thoroughly and mechanistically explain why the NN gets certain digits wrong

# Standard way of thinking about a standard single hidden layer NN

+ There are two (three if you count inputs as neurons) layers of neurons
+ We pass iteratively from one layer to the next
+ Therefore to understand why 

# Diagram of standard neural net

<img src="./neural-net.svg"/>

# Standard layer by layer decomposition

<img src="./neural-net-layer-decomposition.svg" width="500px"/>

# Alternative decomposition

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
