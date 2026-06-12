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

## Changing the default API key

The API key that the deployed site hands out to participants lives in exactly one place: the `"apiKey"` field of `config.json` in this folder (not in the source code — `DEFAULT_SETTINGS` in `src/types/index.ts` deliberately has an empty key).

1. Edit `config.json` (create it first with `cp config.json.example config.json` if it doesn't exist) and set the new key:

   ```json
   {
     "apiKey": "sk-or-v1-..."
   }
   ```

2. Run the deploy script from this folder:

   ```bash
   ./upload.sh
   ```

   This uploads the new `config.json` to the server. Reload the deployed site to confirm; note that existing conversations keep the settings they were created with, so test with a fresh "New Chat".

### SSH key setup (needed for upload.sh)

`upload.sh` rsyncs over SSH to `tarospec_aisap-test-website@ssh.nyc1.nearlyfreespeech.net`. If you get a permission/password prompt that fails, set up an SSH key:

1. Generate a key if you don't already have one (`ls ~/.ssh/*.pub` to check):

   ```bash
   ssh-keygen -t ed25519
   ```

2. Authorize it on NearlyFreeSpeech, either of:
   - Log in to the NearlyFreeSpeech member interface, go to **Profile → Add SSH Key**, and paste the contents of `~/.ssh/id_ed25519.pub`; or
   - If password SSH login works for the site, run:

     ```bash
     ssh-copy-id tarospec_aisap-test-website@ssh.nyc1.nearlyfreespeech.net
     ```

3. Verify with `ssh tarospec_aisap-test-website@ssh.nyc1.nearlyfreespeech.net true` — if that succeeds without a password prompt, `./upload.sh` will work.
