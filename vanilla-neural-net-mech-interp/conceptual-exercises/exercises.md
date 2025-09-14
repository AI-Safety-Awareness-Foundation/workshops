*Exercise*:

> Calculate what the output of this single neuron is if it was given the two
> inputs 2.0 and 1.5 (corresponding respectively to the top input and the bottom
> input).
>
> ![single neuron](./single_neuron_worked.jpeg)

<details>
<summary>Solution</summary>
$\text{ReLU}(2.0 \cdot 0.5 + 1.5 \cdot -0.1 + 1.3) = 2.15$
</details>

*Exercise*:

> Calculate what the output of this single neuron is if it was given the two
> inputs 0.1 and 20 (corresponding respectively to the top input and the bottom
> input).
>
> ![single neuron](./single_neuron_worked.jpeg)

<details>
<summary>Solution</summary>
$\text{ReLU}(0.1 \cdot 0.5 + 5 \cdot -0.1 + 1.3) = \text{ReLU}(-0.65) = 0$
</details>

*Exercise*: 

> Given the following diagram, calculate what the output of the neural net
> should be "layer by layer" in the traditional way
> using inputs 1.5 and -0.8.
>
> ![exercise diagram](./exercise-question.jpeg)

<details>
<summary>Solution</summary>

![solution diagram](./traditional-solution.jpeg)

</details>

*Exercise*: 

> Given the following diagram, calculate what the output of the neural net
> should be in our "key-value" decomposition of a
> vanilla neural net using inputs 1.5 and -0.8
> again.
>
> ![exercise diagram](./exercise-question.jpeg)

<details>
<summary>Solution</summary>

![solution diagram](./key-value-solution.jpeg)

</details>
