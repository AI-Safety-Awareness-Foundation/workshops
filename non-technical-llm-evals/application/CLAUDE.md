# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a manual LLM evaluation interface - a frontend-only TypeScript application that allows non-technical users to evaluate LLM behavior through a chat interface. The key feature is **parallel response sampling**: each user message triggers multiple LLM calls (e.g., 20 times) to assess the propensity of certain behaviors.

### Core Functionality

1. **Parallel Response Display**: Run the same prompt multiple times and display all responses side-by-side to reveal behavioral patterns (e.g., "Would you hurt a hamster?" → 15 "no", 5 "yes")

2. **Automatic Classification**: A second model (or separate instance) acts as a classifier to categorize responses as "good" or "bad", automatically flagging concerning answers. Users define custom classification criteria that the evaluator model uses to assess each response.

3. **Conversation Control**:
   - User-editable system prompts
   - Ability to edit both user and assistant messages
   - Support for conversation "prefixes" (multi-turn conversations that the model continues)

4. **Hard-coded Tool Access**:
   - Mock filesystem (read/write) pre-seeded with test files
   - Mock email tool with API: `emailer(sender-address, recipient-address, subject, body)`

5. **Client-side Only**: No backend - users provide API keys via the frontend, and all LLM calls happen client-side

## Architecture

- **Framework**: React 18 + TypeScript + Vite
- **API Client**: @anthropic-ai/sdk (client-side with dangerouslyAllowBrowser)
- **State Management**: React hooks (useState, useRef)
- **Styling**: Plain CSS with grid layout for responsive design

### Key Files

- `src/services/llmService.ts`: Core orchestration for parallel LLM calls, tool execution, and Anthropic API integration
- `src/mockTools/filesystem.ts`: Mock filesystem pre-seeded with evaluation test files
- `src/mockTools/email.ts`: Mock email service for tracking sent emails
- `src/components/`: React UI components (ConfigPanel, MessageEditor, ResponseGrid, ChatInput)
- `src/App.tsx`: Main application component with state management
- `src/types.ts`: TypeScript type definitions

### Implementation Details

- **Parallel Calls**: `LLMService.runParallelCalls()` uses Promise.all to run N model calls concurrently
- **Tool Use Loop**: Handles Anthropic's tool_use stop reason, executes tools, and continues conversation
- **API Key**: Stored in component state (session-only, never persisted)
- **Mock Tools**: Implemented as TypeScript classes, executed synchronously when model calls them
- **Response Classification**: `LLMService.classifyResponse()` makes a separate LLM call per response with user-defined criteria, returning 'good' or 'bad'. Classifications run in parallel after all responses are collected. The classifier receives the user prompt, assistant response, and all tool calls.

## Development Commands

```bash
npm install          # Install dependencies
npm run dev         # Start development server (default: http://localhost:5173)
npm run build       # Build for production (outputs to dist/)
npm run preview     # Preview production build
```

## Key Design Considerations

- This is for **evaluation**, not production chat - prioritize visibility into model behavior
- Non-technical users are the target audience - the interface must be intuitive
- The parallel sampling feature is the core differentiator from standard chat interfaces
- Tool calls should be logged/displayed so evaluators can see what the model attempted

## UI Reference

See `mockup-of-interface.png` for the interface design (editable via `sketch-of-interface.excalidraw`)
