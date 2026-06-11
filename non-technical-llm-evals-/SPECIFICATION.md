# ChatGPT Clone Specification

A shallow clone of ChatGPT designed to help users understand how prefill jailbreaking attacks work by exposing more of the underlying mechanics.

## Tech Stack

- **Frontend**: React with TypeScript
- **Backend**: None (frontend-only, API calls made directly from browser)
- **Styling**: Plain CSS

## LLM API Configuration

### Endpoints
- **OpenRouter**: Uses OpenRouter API as primary endpoint
- **Custom VLLM**: User-configurable VLLM URL as alternative endpoint

### Configuration Scope
- Per-conversation settings (each conversation can have its own endpoint, API key, and model)

### Model Selection
- Dropdown with common OpenRouter models
- Custom input option for manually typing model IDs

## Core Features

### 1. Streaming Responses
- Responses stream in token-by-token (not waiting for full completion)

### 2. Conversation History Editing
- Users can edit the assistant's previous responses
- Editing creates a new branch (old response is preserved)
- UI will need to handle branch navigation/visualization

### 3. Prefill / Continue Generation
- Inline editing of the last assistant message
- User can type a prefill and have the model continue from there

### 4. Raw Conversation View
- Button to expose the entire conversation as raw text
- Opens in a **side panel**
- **Two tabs**:
  - Plain text with role markers and annotations for thinking tokens and tool calls
  - JSON (actual API format)

### 5. System Prompt
- Ability to modify the system prompt per conversation

## Thinking/Reasoning Tokens

- Hidden behind a collapsible "thinking" note by default
- Clicking expands to show the thinking content
- **Configurable parsing mode** (per-conversation setting):
  - Inline `<think>...</think>` tags in content
  - Separate field in API response object

## Tool Calls

- Hidden behind a collapsible "making tool calls" note by default
- Clicking expands to show:
  - Raw text of the tool call
  - Raw response from the tool

### Initial Tools
- **Calculator**: Simple arithmetic operations (first implementation)

### Future Tools (mock implementations)
- Mock email inbox access
- Mock email sending
- Other tools as needed

## Persistence

- Conversations saved to **localStorage**
- Persists across browser refreshes

## Educational Goal

The primary purpose is to give users a familiar chat interface while exposing the underlying mechanics, ultimately helping them understand how prefill jailbreaking attacks work.
