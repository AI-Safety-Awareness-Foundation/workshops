# %%

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

# %%

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
)

# %%

prompt = """

A farmer has chickens and rabbits. There are 20 heads and 56 legs total.
How many chickens and how many rabbits are there?
"""

stream = client.chat.completions.create(
    model="google/gemini-2.5-pro",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    extra_body={
        "reasoning": {
            "effort": "high"
        }
    },
    stream=True,
)

for event in stream:
    if event.choices[0].delta.reasoning:
        print(f"Reasoning: {event.choices[0].delta.reasoning=}")
    else:
        print(f"Content: {event.choices[0].delta.content}")


# %%

def stream_with_reasoning(prompt: str, model: str = "google/gemini-2.5-pro"):
    """
    Stream a response from a reasoning model, showing both thinking and response.
    
    Args:
        prompt: The user's prompt
        model: The model to use (default: Gemini 2.5 Flash Thinking)
    """
    print(f"Using model: {model}\n")
    print("=" * 80)
    
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True,
        )
        
        in_thinking = False
        thinking_content = []
        response_content = []
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                
                # Check if we're entering or in a thinking block
                if "<thinking>" in content or in_thinking:
                    in_thinking = True
                    thinking_content.append(content)
                    print(content, end="", flush=True)
                    
                    if "</thinking>" in content:
                        in_thinking = False
                        print("\n" + "=" * 80)
                        print("RESPONSE:")
                        print("=" * 80)
                else:
                    # Regular response content
                    response_content.append(content)
                    print(content, end="", flush=True)
        
        print("\n" + "=" * 80)
        
        return {
            "thinking": "".join(thinking_content),
            "response": "".join(response_content)
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# %%

prompt = """
Solve this problem step by step:

A farmer has chickens and rabbits. There are 20 heads and 56 legs total.
How many chickens and how many rabbits are there?
"""

print("PROMPT:")
print(prompt)
print("\n" + "=" * 80)
print("THINKING:")
print("=" * 80)

result = stream_with_reasoning(prompt)

if result:
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Thinking length: {len(result['thinking'])} characters")
    print(f"Response length: {len(result['response'])} characters")