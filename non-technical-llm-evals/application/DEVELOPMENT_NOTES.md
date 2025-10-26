# Development Notes

## Recent Development Sessions (Jan 2025)

### Session Summary
Built a complete manual LLM evaluation interface from scratch with the following major features:
1. Parallel response sampling (1-50 configurable calls)
2. Automatic classification with user-definable criteria
3. Separate model selection for classification vs evaluation
4. Unified conversation interface with prefix builder
5. Mock tools (filesystem, email) for evaluation scenarios

### Technical Decisions Made

#### Classification Architecture
**Decision**: Run classification AFTER all responses are collected, not during.
**Rationale**:
- Prevents classification from blocking response generation
- Allows all responses to be displayed immediately
- Classifications can run in parallel for better performance
- User sees responses faster even if classification is slow

**Implementation**:
```typescript
// In LLMService.runParallelCalls()
const responses = await Promise.all(promises); // Collect all responses first
if (enableClassification) {
  // Then classify in parallel
  return Promise.all(classificationPromises);
}
```

#### Classification Model Selection
**Decision**: Allow separate model for classification independent from evaluation model.
**Rationale**:
- Cost optimization: use Haiku for cheap classification on large batches
- Quality control: use Opus when nuanced safety evaluation needed
- Flexibility: experiment with different evaluator/classifier combinations

#### UI Layout - Unified Conversation View
**Decision**: Move conversation prefix from sidebar to main area, integrate with evaluation input.
**Rationale**:
- Creates natural conversation flow (top to bottom)
- Makes it clearer that evaluation prompt continues the conversation
- Sidebar can focus purely on configuration
- More space for conversation building

**Important**: Evaluation input does NOT clear after submit (intentional design for iteration).

#### Classification Criteria Visibility
**Decision**: Always show classification criteria text box, not conditional on checkbox.
**Rationale**:
- Encourages users to think about evaluation criteria upfront
- Makes it easy to prepare criteria before enabling classification
- Reduces UI jumping when toggling checkbox

#### TypeScript Gotchas
**Issue**: Anthropic SDK tool definitions require literal types, not inferred strings.
**Solution**: Use `as const` for nested properties but NOT for the outer array.
```typescript
type: 'object' as const  // ✓ Correct
type: 'string' as const  // ✓ Correct
] as const;  // ✗ Makes entire array readonly - causes type errors
```

### Mock Tools Implementation

#### File System Pre-seeding
Located in `src/mockTools/filesystem.ts` constructor.

Files included for evaluation scenarios:
- `/readme.txt` - Benign welcome message
- `/secrets/password.txt` - Test credentials (for testing data access behavior)
- `/documents/personal_info.txt` - PII data (SSN, address)
- `/code/script.py` - Dangerous deletion script
- `/emails/draft.txt` - Potentially sensitive email draft
- `/notes/todo.txt` - Benign personal notes

**Design principle**: Mix of benign and sensitive files to test model caution.

#### Email Service
Located in `src/mockTools/email.ts`.

Tracks all sent emails in memory with timestamp. Does not actually send emails (all mock).

### Component Communication Patterns

#### App.tsx → ConversationView
```typescript
<ConversationView
  messages={prefixMessages}           // Pass down conversation state
  onMessagesChange={setPrefixMessages} // Callback to update state
  onEvaluate={handleSendMessage}      // Callback for evaluation submit
  isLoading={isLoading}               // Disable UI during loading
/>
```

**Key insight**: ConversationView manages its own local state (editingId, evaluationInput) but conversation messages live in parent.

#### State Management Philosophy
- Configuration state lives in App.tsx (single source of truth)
- UI state lives in components (editing states, local inputs)
- API key never persists (session-only in memory)
- All mock tool state resets on page reload (intentional - evaluation session scope)

### CSS Architecture

#### Response Card Badges
Using `::before` pseudo-elements for classification badges:
```css
.response-card.bad::before {
  content: "⚠ BAD / MALICIOUS";
  /* Badge styling */
}
```

**Advantage**: Badges are purely presentational, not in DOM. Keeps component code clean.

#### Layout Strategy
- Flexbox for main layout (sidebar + main content)
- CSS Grid for response cards (auto-fill with minmax)
- Fixed sidebar width (350px)
- Vertical split in main content (conversation section + results section)

### Performance Considerations

#### Parallel Response Generation
- Uses `Promise.all()` for concurrent API calls
- Progress tracking via callback function
- No throttling or rate limiting (assumes API handles this)

**Potential issue**: Very high parallel call counts (40-50) may hit rate limits.

#### Classification Performance
- Max 10 tokens per classification (very fast)
- Runs in parallel after response collection
- Error handling: defaults to 'good' on classification failure

### Future Development Notes

#### Known Limitations
1. No response persistence - results lost on page reload
2. No export functionality - can't save evaluation results
3. Hard-coded tool definitions - can't add custom tools via UI
4. No statistical summary - user must manually count good/bad responses
5. No response filtering - all responses always shown

#### Potential Improvements
1. **Export**: Add CSV/JSON export of responses with classifications
2. **Statistics**: Show summary bar (e.g., "3/20 bad responses, 15% propensity")
3. **Filtering**: Toggle to show only bad responses, only good responses
4. **Custom Tools**: UI for defining additional tools beyond the 4 built-in ones
5. **Conversation History**: Save/load conversation prefixes
6. **Response Comparison**: Side-by-side diff view for analyzing variations

#### Code Quality Notes
- No tests written (manual testing only)
- Type safety via TypeScript (strict mode enabled)
- No linting configuration (relies on defaults)
- No error boundaries (errors will crash app)

### Dependencies

Key packages:
- `@anthropic-ai/sdk`: Official Anthropic API client
- `react` + `react-dom`: UI framework
- `vite`: Build tool and dev server
- `typescript`: Type checking

**Important**: Using `dangerouslyAllowBrowser: true` because this is client-side only. In production with a backend, this flag should NOT be used.

### Environment Setup

No environment variables needed. API key provided by user at runtime.

Dev server: `npm run dev` → http://localhost:5173
Build: `npm run build` → outputs to `dist/`

### Git History Highlights

Key commits:
- `e65721e`: Initial implementation (parallel calls, mock tools, basic UI)
- `10490be`: Added classification with user-definable criteria
- `2f6fe9d`: Separate model selection for classification
- `170b282`: UI redesign with unified conversation view
- `d3a8c3d`: Made evaluation input persistent

## Questions for Future Developers

1. Should we add rate limiting for parallel calls?
2. Should classification criteria have templates/presets?
3. Should we support other LLM providers (OpenAI, etc.)?
4. Should we add conversation templates for common evaluation scenarios?
5. Should mock filesystem be user-editable via UI?

## Common Development Tasks

### Adding a New Mock Tool
1. Update `LLMService.getTools()` with tool definition
2. Add case to `LLMService.executeTool()` switch statement
3. Consider creating new mock service class if complex state needed
4. Update USAGE.md with tool documentation

### Changing Classification Logic
Located in `LLMService.classifyResponse()`. Currently does simple keyword matching for "BAD" in response. Could be enhanced to:
- Use structured output
- Multi-class classification (not just good/bad)
- Confidence scores
- Multiple classification dimensions

### Modifying UI Layout
Main layout in `App.css`:
- `.app-layout`: Overall flex container
- `.sidebar`: Fixed width, scrollable
- `.main-content`: Flex column, two sections
- `.conversation-section`: Auto height, white background
- `.results-section`: Flex 1 (takes remaining space)
