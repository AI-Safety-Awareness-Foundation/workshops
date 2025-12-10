// Thinking token format configuration
export type ThinkingTokenFormat = 'inline-tags' | 'separate-field';

// API endpoint type
export type EndpointType = 'openrouter' | 'vllm';

// Per-conversation settings
export interface ConversationSettings {
  endpointType: EndpointType;
  apiKey: string;
  vllmUrl: string;
  model: string;
  systemPrompt: string;
  thinkingTokenFormat: ThinkingTokenFormat;
}

// Tool call types
export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface ToolResult {
  toolCallId: string;
  result: string;
}

// Message types
export interface ThinkingContent {
  type: 'thinking';
  content: string;
}

export interface TextContent {
  type: 'text';
  content: string;
}

export interface ToolCallContent {
  type: 'tool_call';
  toolCall: ToolCall;
  result?: ToolResult;
}

export type MessageContent = ThinkingContent | TextContent | ToolCallContent;

export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: MessageContent[];
  // Raw content string for display/editing
  rawContent: string;
  // For branching: ID of the parent message (null for first message)
  parentId: string | null;
  // For branching: IDs of child messages (branches)
  childIds: string[];
  // Timestamp
  createdAt: number;
  // For tool messages: the tool call ID this is responding to
  toolCallId?: string;
}

// A conversation is a tree of messages
// We track the "active path" through the tree
export interface Conversation {
  id: string;
  title: string;
  settings: ConversationSettings;
  // All messages in the conversation (stored as a map for easy lookup)
  messages: Record<string, Message>;
  // Root message ID (usually the system prompt or first user message)
  rootMessageId: string | null;
  // Currently active leaf message ID (the end of the current branch)
  activeLeafId: string | null;
  createdAt: number;
  updatedAt: number;
}

// App state
export interface AppState {
  conversations: Record<string, Conversation>;
  activeConversationId: string | null;
}

// Default settings
export const DEFAULT_SETTINGS: ConversationSettings = {
  endpointType: 'openrouter',
  apiKey: '',
  vllmUrl: 'http://localhost:8000',
  model: 'anthropic/claude-3.5-sonnet',
  systemPrompt: 'You are a helpful assistant.',
  thinkingTokenFormat: 'inline-tags',
};

// Common OpenRouter models
export const COMMON_MODELS = [
  { id: 'anthropic/claude-sonnet-4', name: 'Claude Sonnet 4' },
  { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet' },
  { id: 'anthropic/claude-3.5-haiku', name: 'Claude 3.5 Haiku' },
  { id: 'openai/gpt-4o', name: 'GPT-4o' },
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini' },
  { id: 'google/gemini-pro-1.5', name: 'Gemini Pro 1.5' },
  { id: 'deepseek/deepseek-r1', name: 'DeepSeek R1' },
  { id: 'meta-llama/llama-3.1-405b-instruct', name: 'Llama 3.1 405B' },
];
