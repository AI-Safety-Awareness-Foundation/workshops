# %%

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_KEY,
)

# %%

completion = client.chat.completions.create(
  model="openai/gpt-4o",
  messages=[
    {
      "role": "user",
      "content": "What is the meaning of life?"
    }
  ],
  logprobs=True,
  top_logprobs=5
)

# %%
import math
import copy

def get_token_probabilities(completion):
  probabilities = []
  for choice in completion.model_dump()['choices']:
    new_logprobs = copy.deepcopy(choice['logprobs']['content'])
    for logprob_item in new_logprobs:
      for candidate_logprob_item in logprob_item['top_logprobs']:
        candidate_logprob_item['prob'] = math.exp(candidate_logprob_item['logprob'])
      logprob_item['prob'] = math.exp(logprob_item['logprob'])
    probabilities.append(new_logprobs)
  return probabilities

get_token_probabilities(completion)

# %%

print(f"{completion.model_dump()=}")

print(completion.choices[0].message.content)

