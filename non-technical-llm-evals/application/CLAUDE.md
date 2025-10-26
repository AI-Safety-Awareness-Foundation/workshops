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

- `src/services/llmService.ts`: Core orchestration for parallel LLM calls, tool execution, classification, and Anthropic API integration
- `src/mockTools/filesystem.ts`: Mock filesystem pre-seeded with evaluation test files (passwords, PII, dangerous scripts, etc.)
- `src/mockTools/email.ts`: Mock email service for tracking sent emails
- `src/components/ConversationView.tsx`: Unified conversation interface (prefix builder + evaluation input)
- `src/components/ConfigPanel.tsx`: Configuration sidebar (API key, models, system prompt, classification settings)
- `src/components/ResponseGrid.tsx`: Parallel response display with classification badges
- `src/App.tsx`: Main application component with state management
- `src/types.ts`: TypeScript type definitions (Message, ParallelResponse, EvalConfig, etc.)

### Implementation Details

- **Parallel Calls**: `LLMService.runParallelCalls()` uses Promise.all to run N model calls concurrently (configurable 1-50)
- **Tool Use Loop**: Handles Anthropic's tool_use stop reason, executes mock tools synchronously, continues conversation with tool results
- **API Key**: Stored in component state (session-only, never persisted to disk)
- **Mock Tools**: Implemented as TypeScript classes with in-memory state, tools include: read_file, write_file, list_directory, emailer
- **Response Classification**:
  - `LLMService.classifyResponse()` makes separate LLM call per response with user-defined criteria
  - Classifications run in parallel AFTER all responses collected (not during)
  - Classifier receives: user prompt + assistant response + all tool calls (with results)
  - Returns 'good' or 'bad' based on keyword matching ("BAD" in response)
  - Separate model can be selected for classification (defaults to same as evaluation model)
  - Max 10 tokens per classification for efficiency
- **UI Layout**:
  - Sidebar: Configuration only (350px fixed width)
  - Main area split vertically: conversation section (top) + results section (bottom)
  - Conversation section: unified view with prefix messages + evaluation input
  - Results section: scrollable grid of parallel responses

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
- Classification criteria is ALWAYS visible (not hidden behind checkbox) to encourage thoughtful evaluation design
- Evaluation prompt persists after submission for easy re-running and iteration

## Component Architecture

### ConversationView
- Combines conversation prefix builder with evaluation input in single component
- Manages local state for message editing (editingId) and evaluation input
- Messages editable inline (click to edit, blur to save)
- Color-coded borders: blue (user), green (assistant)
- Empty state encourages direct evaluation or building prefix
- Evaluation input does NOT clear on submit (intentional for iteration)

### ConfigPanel
- All configuration in left sidebar
- Classification criteria visible always (not conditional on checkbox)
- Classification model selector only shows when classification enabled
- Uses controlled components pattern for all inputs

### ResponseGrid
- Grid layout with auto-fill columns (minmax 300px)
- Visual indicators via CSS pseudo-elements (::before)
  - Good: green border + "✓ GOOD" badge
  - Bad: red border with glow + "⚠ BAD / MALICIOUS" badge
- Tool calls displayed in expandable sections within each response card
- Progress indicator during loading

## Important Implementation Notes

### Classification Flow
1. User submits evaluation with classification enabled
2. All N parallel responses collected first
3. THEN classification runs in parallel for all responses
4. This prevents classification from blocking response generation
5. Error responses are NOT classified (skip classification for errors)

### Tool Execution
- Tools execute synchronously and immediately return results
- No actual file I/O or network calls (all in-memory)
- Tool results are strings, returned to model in next turn
- Mock filesystem persists during session but resets on page reload
- Email history accumulates during session

### State Management
- prefixMessages: Array of conversation setup messages
- evaluationInput: Current text in evaluation textarea (persists after submit)
- responses: Current batch of parallel responses with classifications
- config: All configuration settings (models, prompts, API key, etc.)
- isLoading: Boolean to disable UI during evaluation

### Model Selection
- Evaluation model: Used for generating all parallel responses
- Classification model: Used for classifying responses (can be different)
- Both default to Claude 3.5 Sonnet
- Options: Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku

## Future Enhancement Ideas
- Export results to CSV/JSON
- Response filtering by classification (show only bad, etc.)
- Statistical summary (e.g., "3/20 responses classified as bad")
- Custom tool definitions (currently hard-coded)
- Tool call aggregation across responses
- Response comparison view (side-by-side diff)
