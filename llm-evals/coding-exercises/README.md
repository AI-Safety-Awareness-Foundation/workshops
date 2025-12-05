# Completing these exercises

Make sure you have your `.env` file set up correctly! You'll need an OpenRouter
API key and an instance of VLLM running to be able to serve up some open
weights LLMs (that we can run all sorts of safety experiments without having to
worry about running afoul of usage policies).

We recommend running VLLM with Modal, which makes it very easy to spin up a
server with an appropriately beefy GPU. If you have the Modal commandline
program installed, you can just run `modal deploy modal_vllm.py` where
`modal_vllm.py` is a file in this directory that sets up a Qwen model.

## Doing the exercises

You can either run the exercises locally (using either `exercises.py` if you
would like to work with a direct Python script, or `exercises.ipynb` if you'd
like to use Jupyter notebook).

If you want to get started immediately without setting up a local developer, you
can always open this in Google Colab with
[https://colab.research.google.com/github/AI-Safety-Awareness-Foundation/workshops/blob/master/llm-evals/coding-exercises/exercises.ipynb](https://colab.research.google.com/github/AI-Safety-Awareness-Foundation/workshops/blob/master/llm-evals/coding-exercises/exercises.ipynb).

You can always find solutions to the exercises either at `solutions.py` and
`solutions.ipynb` and likewise can run the solutions in Google Colab at
[https://colab.research.google.com/github/AI-Safety-Awareness-Foundation/workshops/blob/master/llm-evals/coding-exercises/solutions.ipynb](https://colab.research.google.com/github/AI-Safety-Awareness-Foundation/workshops/blob/master/llm-evals/coding-exercises/solutions.ipynb).
