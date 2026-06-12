# LLM Evals Chat Playground

A browser-based chat interface for the non-technical LLM evals workshop. It looks like a ChatGPT-style app, but exposes the knobs that workshop participants need to poke at model behavior directly: system prompts, prefilled assistant responses, editable messages (including tool results), conversation branching, and a mock email inbox for agentic scenarios.

Built with React + TypeScript + Vite. All state lives in the browser's localStorage; there is no backend — API calls go straight from the browser to the model provider.

## Features

- **Endpoints**: OpenRouter (default) or a self-hosted vLLM server, configurable per conversation along with model, API key, and system prompt.
- **Branching conversations**: editing a user, assistant, or tool message creates a new branch; you can switch between branches at any point in the tree.
- **Prefill**: start an assistant response with text of your choosing and let the model continue from it.
- **Thinking tokens**: parses `<think>...</think>` blocks inline and renders them as collapsible thinking sections.
- **Tools**: calculator, read-inbox, and send-email (mock) tools the model can call. The inbox is editable per conversation via the inbox editor, and "sent" emails appear as toast notifications.
- **Raw view**: see the conversation as plain text or as the JSON request that would be sent to the API.

## Development

```bash
npm install
npm run dev
```

## Default settings and config.json

Built-in defaults live in `DEFAULT_SETTINGS` in `src/types/index.ts`. At startup the app fetches `/config.json` and merges any keys it contains over those defaults (see `src/utils/config.ts`), so new conversations pick up the deployed values — most importantly the OpenRouter API key, which is intentionally not committed.

`config.json` is gitignored. To set up local overrides:

```bash
cp config.json.example config.json
# then edit config.json and fill in the API key
```

Any subset of the `ConversationSettings` fields (`apiKey`, `model`, `endpointType`, `systemPrompt`, ...) may be included. Note that whatever you put in `config.json` is served publicly by the deployed site, so only use keys you are willing to expose to workshop participants.

## Deployment

```bash
./upload.sh
```

This builds the app and rsyncs `dist/` to the NearlyFreeSpeech host, then uploads `config.json` alongside it if present. If `config.json` is missing it warns and deploys without it, leaving whatever `config.json` is already on the server (if any) in place.
