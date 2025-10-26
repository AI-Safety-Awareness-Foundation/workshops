# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a manual LLM evaluation interface - a frontend-only TypeScript application that allows non-technical users to evaluate LLM behavior through a chat interface. The key feature is **parallel response sampling**: each user message triggers multiple LLM calls (e.g., 20 times) to assess the propensity of certain behaviors.

### Core Functionality

1. **Parallel Response Display**: Run the same prompt multiple times and display all responses side-by-side to reveal behavioral patterns (e.g., "Would you hurt a hamster?" → 15 "no", 5 "yes")

2. **Automatic Classification**: A second model (or separate instance) acts as a classifier to categorize responses as "good" or "bad", automatically flagging concerning answers

3. **Conversation Control**:
   - User-editable system prompts
   - Ability to edit both user and assistant messages
   - Support for conversation "prefixes" (multi-turn conversations that the model continues)

4. **Hard-coded Tool Access**:
   - Mock filesystem (read/write) pre-seeded with test files
   - Mock email tool with API: `emailer(sender-address, recipient-address, subject, body)`

5. **Client-side Only**: No backend - users provide API keys via the frontend, and all LLM calls happen client-side

## Architecture Notes

- **Framework**: TypeScript-based (no specific framework chosen yet)
- **API Key Handling**: Client-side storage and usage
- **Mock Tools**: Need to implement mock filesystem and email service that can be exposed to the LLM
- **Multi-response Orchestration**: Need system to trigger N parallel LLM calls and aggregate results
- **Response Classification**: Integration point for a classifier model to categorize responses

## Development Commands

Currently minimal dependencies installed. When the application is scaffolded, commands will likely include:

```bash
npm install          # Install dependencies
npm run dev         # Start development server
npm run build       # Build for production
npm run test        # Run tests (when implemented)
```

## Key Design Considerations

- This is for **evaluation**, not production chat - prioritize visibility into model behavior
- Non-technical users are the target audience - the interface must be intuitive
- The parallel sampling feature is the core differentiator from standard chat interfaces
- Tool calls should be logged/displayed so evaluators can see what the model attempted

## UI Reference

See `mockup-of-interface.png` for the interface design (editable via `sketch-of-interface.excalidraw`)
