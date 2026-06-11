I'd like to build a shallow clone of the ChatGPT that does a couple of things
differently:

1. Uses OpenRouter as well as a user-configurable VLLM URL as the LLM API
   endpoint instead of OpenAI's servers.
2. Lets me edit the assistant's previous answers in the conversation history
   and re-run the conversation.
3. Also lets me prefill the assistant's last response and have it continue
   generation from there.
4. Has the ability to expose the entire conversation as a single raw text,
   e.g. by clicking on a button we can see the entire conversation as raw text
   just marked up with different roles and 
5. Has the ability to modify the system prompt

A few notes on some things it should keep from the usual ChatGPT interface:

1. Responses should stream in token by token (instead of e.g. waiting until the
   entire turn is finished before showing the results)
2. By default it should hide thinking/reasoning tokens behind a "thinking"
   note.
    1. Note that thinking/reasoning tokens should still be visible by clicking
       on the "thinking" note in the same way that this holds for ChatGPT and
       Claude interfaces.
    2. The 
3. Likewise tool calls (and responses) should be hidden behind a "making tool
   calls" note.
    1. By clicking on it we should be able to see the raw text that goes into a
       tool call and then the raw response from the client that fills in the
       tool call response.

The goal here is to ultimately give users a familiar interface that will
ultimately explain to them how prefill jail-breaking attacks work.

Please ask a lot of clarifying questions before going and doing an
implementation to remove ambiguity.
