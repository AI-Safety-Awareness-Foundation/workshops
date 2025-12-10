import { v4 as uuidv4 } from 'uuid';
import type { Conversation, Message, MessageContent, AppState } from '../types';
import { DEFAULT_SETTINGS } from '../types';

// Generate unique IDs
export function generateId(): string {
  return uuidv4();
}

// Create a new conversation
export function createConversation(): Conversation {
  const id = generateId();
  const now = Date.now();
  return {
    id,
    title: 'New Chat',
    settings: { ...DEFAULT_SETTINGS },
    messages: {},
    rootMessageId: null,
    activeLeafId: null,
    createdAt: now,
    updatedAt: now,
  };
}

// Create a new message
export function createMessage(
  role: 'user' | 'assistant' | 'system',
  rawContent: string,
  parentId: string | null = null
): Message {
  return {
    id: generateId(),
    role,
    content: [{ type: 'text', content: rawContent }],
    rawContent,
    parentId,
    childIds: [],
    createdAt: Date.now(),
  };
}

// Get the active message path (from root to active leaf)
export function getActivePath(conversation: Conversation): Message[] {
  if (!conversation.activeLeafId) return [];

  const path: Message[] = [];
  let currentId: string | null = conversation.activeLeafId;

  // Build path from leaf to root
  while (currentId !== null) {
    const msg: Message | undefined = conversation.messages[currentId];
    if (!msg) break;
    path.unshift(msg);
    currentId = msg.parentId;
  }

  return path;
}

// Add a message to the conversation
export function addMessage(
  conversation: Conversation,
  message: Message
): Conversation {
  const updatedConversation = { ...conversation };
  updatedConversation.messages = {
    ...conversation.messages,
    [message.id]: message,
  };

  // Update parent's childIds if there's a parent
  if (message.parentId && conversation.messages[message.parentId]) {
    const parent = conversation.messages[message.parentId];
    updatedConversation.messages[message.parentId] = {
      ...parent,
      childIds: [...parent.childIds, message.id],
    };
  }

  // Set as root if no root exists
  if (!conversation.rootMessageId) {
    updatedConversation.rootMessageId = message.id;
  }

  // Update active leaf
  updatedConversation.activeLeafId = message.id;
  updatedConversation.updatedAt = Date.now();

  return updatedConversation;
}

// Update a message in the conversation
export function updateMessage(
  conversation: Conversation,
  messageId: string,
  updates: Partial<Message>
): Conversation {
  if (!conversation.messages[messageId]) return conversation;

  return {
    ...conversation,
    messages: {
      ...conversation.messages,
      [messageId]: {
        ...conversation.messages[messageId],
        ...updates,
      },
    },
    updatedAt: Date.now(),
  };
}

// Parse thinking tokens from content (inline tags)
export function parseInlineThinking(content: string): MessageContent[] {
  const result: MessageContent[] = [];
  const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
  let lastIndex = 0;
  let match;

  while ((match = thinkRegex.exec(content)) !== null) {
    // Add text before thinking block
    if (match.index > lastIndex) {
      const textBefore = content.slice(lastIndex, match.index).trim();
      if (textBefore) {
        result.push({ type: 'text', content: textBefore });
      }
    }

    // Add thinking block
    result.push({ type: 'thinking', content: match[1].trim() });
    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < content.length) {
    const remaining = content.slice(lastIndex).trim();
    if (remaining) {
      result.push({ type: 'text', content: remaining });
    }
  }

  // If no thinking blocks found, return the whole content as text
  if (result.length === 0) {
    result.push({ type: 'text', content });
  }

  return result;
}

// Generate plain text representation of conversation
export function conversationToPlainText(conversation: Conversation): string {
  const path = getActivePath(conversation);
  let text = '';

  // Add system prompt if present
  if (conversation.settings.systemPrompt) {
    text += `[SYSTEM]:\n${conversation.settings.systemPrompt}\n\n`;
  }

  for (const message of path) {
    const role = message.role.toUpperCase();
    text += `[${role}]:\n`;

    for (const content of message.content) {
      if (content.type === 'thinking') {
        text += `<thinking>\n${content.content}\n</thinking>\n`;
      } else if (content.type === 'text') {
        text += `${content.content}\n`;
      } else if (content.type === 'tool_call') {
        text += `<tool_call name="${content.toolCall.name}">\n${content.toolCall.arguments}\n</tool_call>\n`;
        if (content.result) {
          text += `<tool_result>\n${content.result.result}\n</tool_result>\n`;
        }
      }
    }

    text += '\n';
  }

  return text.trim();
}

// Generate JSON representation of conversation (API format)
export function conversationToJSON(conversation: Conversation): string {
  const path = getActivePath(conversation);
  const messages: Array<{ role: string; content: string }> = [];

  // Add system prompt
  if (conversation.settings.systemPrompt) {
    messages.push({
      role: 'system',
      content: conversation.settings.systemPrompt,
    });
  }

  for (const message of path) {
    messages.push({
      role: message.role,
      content: message.rawContent,
    });
  }

  return JSON.stringify(
    {
      model: conversation.settings.model,
      messages,
    },
    null,
    2
  );
}

// Storage keys
const STORAGE_KEY = 'chatgpt-clone-state';

// Save state to localStorage
export function saveState(state: AppState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    console.error('Failed to save state:', e);
  }
}

// Load state from localStorage
export function loadState(): AppState | null {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      return JSON.parse(stored);
    }
  } catch (e) {
    console.error('Failed to load state:', e);
  }
  return null;
}

// Generate conversation title from first user message
export function generateTitle(message: string): string {
  const trimmed = message.trim();
  if (trimmed.length <= 30) return trimmed;
  return trimmed.slice(0, 30) + '...';
}
