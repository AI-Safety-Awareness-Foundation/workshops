# Non-Technical LLM Evals Workshop (Take 2)

Materials for a workshop that teaches LLM evaluations to participants without a programming background. The centerpiece is a browser-based chat playground (`app/`) that exposes the mechanics chat products normally hide — system prompts, assistant prefill, message editing, branching, and tool calls — so participants can probe model behavior and build simple evaluations by hand.

## Contents

- **`app/`** — the chat playground itself (React + TypeScript + Vite). See `app/README.md` for features, development, and deployment instructions.
- **`INITIAL_INSTRUCTIONS.md`** — the original prompt describing what the app should do.
- **`SPECIFICATION.md`** — the fleshed-out spec for the app: a shallow ChatGPT clone designed to demonstrate how prefill jailbreaking attacks work.
- **`INSTRUCTIONS_FOR_INSPECT_LIKE_TOOL.md`** — plan for the next stage: a frontend recreation of UK AISI's [Inspect](https://inspect.aisi.org.uk/) framework (samples / solvers / scorers) so non-programmers can build and run three evals: arithmetic without tools, arithmetic with a calculator tool, and a self-preservation eval using a mock email inbox.
- **`EMAIL_TOOL_EMAIL_FORMAT.md`** — design notes and data format for the mock email inbox tool.
- **`example_email_inbox.xml`** — a small sample inbox for the email tool.
- **`blackmail_email_inbox.xml`** — inbox data for the self-preservation/blackmail scenario (in the style of Anthropic's agentic misalignment setup).
- **`inspect_arithmetic_eval.py`**, **`inspect_arithmetic_tool_call_eval.py`**, **`inspect_email_blackmail_eval.py`** — reference implementations of the three target evals in the real (Python) Inspect framework.
- **`.env`** (gitignored) — `OPENROUTER_API_KEY` and `VLLM_ENDPOINT` for running the Python evals.
