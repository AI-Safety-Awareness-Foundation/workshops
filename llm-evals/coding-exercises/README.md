# Completing these exercises

Make sure you have your `.env` file set up correctly! You'll need an OpenRouter
API key and an instance of VLLM running to be able to serve up some open
weights LLMs (that we can run all sorts of safety experiments without having to
worry about running afoul of usage policies).

We recommend running VLLM with Modal, which makes it very easy to spin up a
server with an appropriately beefy GPU. If you have the Modal commandline
program installed, you can just run `modal deploy modal_vllm.py` where
`modal_vllm.py` is a file in this directory that sets up a Qwen model.
