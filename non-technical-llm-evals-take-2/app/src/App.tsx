import { useState, useEffect, useCallback } from 'react';
import type { AppState, Conversation, MessageContent, ToolCall } from './types';
import {
  createConversation,
  createMessage,
  createToolMessage,
  addMessage,
  updateMessage,
  getActivePath,
  saveState,
  loadState,
  generateTitle,
  parseInlineThinking,
} from './utils/helpers';
import { TOOL_DEFINITIONS, executeToolCall } from './utils/tools';
import Sidebar from './components/Sidebar';
import ChatView from './components/ChatView';
import SettingsPanel from './components/SettingsPanel';
import RawPanel from './components/RawPanel';

function App() {
  const [state, setState] = useState<AppState>(() => {
    const loaded = loadState();
    return loaded || { conversations: {}, activeConversationId: null };
  });

  const [showSettings, setShowSettings] = useState(false);
  const [showRawPanel, setShowRawPanel] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  // Save state to localStorage whenever it changes
  useEffect(() => {
    saveState(state);
  }, [state]);

  const activeConversation = state.activeConversationId
    ? state.conversations[state.activeConversationId]
    : null;

  const handleNewChat = useCallback(() => {
    const newConversation = createConversation();
    setState((prev) => ({
      conversations: {
        ...prev.conversations,
        [newConversation.id]: newConversation,
      },
      activeConversationId: newConversation.id,
    }));
  }, []);

  const handleSelectConversation = useCallback((id: string) => {
    setState((prev) => ({
      ...prev,
      activeConversationId: id,
    }));
  }, []);

  const handleDeleteConversation = useCallback((id: string) => {
    setState((prev) => {
      const { [id]: _, ...remaining } = prev.conversations;
      const newActiveId =
        prev.activeConversationId === id
          ? Object.keys(remaining)[0] || null
          : prev.activeConversationId;
      return {
        conversations: remaining,
        activeConversationId: newActiveId,
      };
    });
  }, []);

  const handleUpdateConversation = useCallback((conversation: Conversation) => {
    setState((prev) => ({
      ...prev,
      conversations: {
        ...prev.conversations,
        [conversation.id]: conversation,
      },
    }));
  }, []);

  const handleSendMessage = useCallback(
    async (content: string, prefill?: string) => {
      if (!activeConversation || isStreaming) return;

      // Create user message
      const userMessage = createMessage(
        'user',
        content,
        activeConversation.activeLeafId
      );

      let updatedConversation = addMessage(activeConversation, userMessage);

      // Update title if this is the first user message
      if (Object.keys(activeConversation.messages).length === 0) {
        updatedConversation = {
          ...updatedConversation,
          title: generateTitle(content),
        };
      }

      // Create assistant message placeholder
      const assistantMessage = createMessage(
        'assistant',
        prefill || '',
        userMessage.id
      );
      updatedConversation = addMessage(updatedConversation, assistantMessage);

      handleUpdateConversation(updatedConversation);

      // Start streaming response
      setIsStreaming(true);

      try {
        await streamResponse(
          updatedConversation,
          assistantMessage.id,
          prefill
        );
      } catch (error) {
        console.error('Streaming error:', error);
        // Update message with error
        setState((prev) => {
          const conv = prev.conversations[updatedConversation.id];
          if (!conv) return prev;
          return {
            ...prev,
            conversations: {
              ...prev.conversations,
              [conv.id]: updateMessage(conv, assistantMessage.id, {
                rawContent: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                content: [
                  {
                    type: 'text',
                    content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                  },
                ],
              }),
            },
          };
        });
      } finally {
        setIsStreaming(false);
      }
    },
    [activeConversation, isStreaming, handleUpdateConversation]
  );

  const streamResponse = async (
    conversation: Conversation,
    messageId: string,
    prefill?: string
  ) => {
    const { settings } = conversation;
    const messages = getActivePath(conversation);
    console.log('[streamResponse] Active path message IDs:', messages.map(m => ({ id: m.id, role: m.role, content: m.rawContent.substring(0, 50) })));
    console.log('[streamResponse] Target messageId:', messageId);

    // Build messages array for API
    type ApiMessage = { role: string; content: string | null; tool_calls?: unknown[]; tool_call_id?: string };
    const apiMessages: ApiMessage[] = [];

    if (settings.systemPrompt) {
      apiMessages.push({ role: 'system', content: settings.systemPrompt });
    }

    // Add all messages except the empty assistant placeholder
    for (const msg of messages) {
      if (msg.id === messageId) {
        // If we have a prefill, add it as the start of assistant response
        if (prefill) {
          apiMessages.push({ role: 'assistant', content: prefill });
        }
        break;
      }
      apiMessages.push({ role: msg.role, content: msg.rawContent });
    }

    const endpoint =
      settings.endpointType === 'openrouter'
        ? 'https://openrouter.ai/api/v1/chat/completions'
        : `${settings.vllmUrl}/v1/chat/completions`;

    // Track accumulated tool calls during streaming
    const accumulatedToolCalls: Map<number, { id: string; name: string; arguments: string }> = new Map();
    let fullContent = prefill || '';
    let hasToolCalls = false;

    console.log('[streamResponse] Starting with prefill:', prefill);
    console.log('[streamResponse] apiMessages:', JSON.stringify(apiMessages, null, 2));

    // Helper to update a message's content
    const updateMessageContent = (targetMessageId: string, text: string, pendingToolCalls: Array<{ id: string; name: string; arguments: string }>) => {
      setState((prev) => {
        const conv = prev.conversations[conversation.id];
        if (!conv) return prev;

        // Parse current text content based on thinking token format
        const currentTextContent: MessageContent[] =
          settings.thinkingTokenFormat === 'inline-tags'
            ? parseInlineThinking(text)
            : text ? [{ type: 'text' as const, content: text }] : [];

        // Add pending tool call content (no results yet)
        const pendingToolContent: MessageContent[] = pendingToolCalls.map((tc) => ({
          type: 'tool_call' as const,
          toolCall: { id: tc.id, name: tc.name, arguments: tc.arguments },
        }));

        const allContent = [...currentTextContent, ...pendingToolContent];

        return {
          ...prev,
          conversations: {
            ...prev.conversations,
            [conv.id]: updateMessage(conv, targetMessageId, {
              rawContent: text,
              content: allContent.length > 0 ? allContent : [{ type: 'text', content: '' }],
            }),
          },
        };
      });
    };

    // Helper to stream a request and update a specific message
    const streamToMessage = async (msgs: ApiMessage[], targetMessageId: string): Promise<void> => {
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${settings.apiKey}`,
          ...(settings.endpointType === 'openrouter' && {
            'HTTP-Referer': window.location.origin,
            'X-Title': 'ChatGPT Clone',
          }),
        },
        body: JSON.stringify({
          model: settings.model,
          messages: msgs,
          stream: true,
          tools: TOOL_DEFINITIONS,
        }),
      });

      if (!response.ok) {
        const error = await response.text();
        throw new Error(`API error: ${response.status} - ${error}`);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No response body');

      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              const choice = parsed.choices?.[0];

              // Handle text content
              const delta = choice?.delta?.content;
              if (delta) {
                fullContent += delta;
                console.log('[streamResponse] Delta received, fullContent now:', fullContent);
                updateMessageContent(targetMessageId, fullContent, []);
              }

              // Handle tool calls
              const toolCallDeltas = choice?.delta?.tool_calls;
              if (toolCallDeltas) {
                hasToolCalls = true;
                for (const tc of toolCallDeltas) {
                  const existing = accumulatedToolCalls.get(tc.index) || { id: '', name: '', arguments: '' };
                  if (tc.id) existing.id = tc.id;
                  if (tc.function?.name) existing.name = tc.function.name;
                  if (tc.function?.arguments) existing.arguments += tc.function.arguments;
                  accumulatedToolCalls.set(tc.index, existing);
                }

                // Update UI with accumulated tool calls
                const toolCalls = Array.from(accumulatedToolCalls.values());
                updateMessageContent(targetMessageId, fullContent, toolCalls);
              }
            } catch {
              // Skip invalid JSON
            }
          }
        }
      }
    };

    // Make initial request, streaming to the first assistant message
    await streamToMessage(apiMessages, messageId);

    // If there were tool calls, execute them and continue
    if (hasToolCalls && accumulatedToolCalls.size > 0) {
      const toolCalls = Array.from(accumulatedToolCalls.values());

      // Finalize the first assistant message with tool calls (including any thinking)
      const contentBeforeTools: MessageContent[] =
        settings.thinkingTokenFormat === 'inline-tags'
          ? parseInlineThinking(fullContent)
          : fullContent ? [{ type: 'text' as const, content: fullContent }] : [];

      // Add tool calls to the first assistant message
      const toolCallContent: MessageContent[] = toolCalls.map((tc) => ({
        type: 'tool_call' as const,
        toolCall: { id: tc.id, name: tc.name, arguments: tc.arguments },
      }));

      // Update first assistant message
      setState((prev) => {
        const conv = prev.conversations[conversation.id];
        if (!conv) return prev;

        return {
          ...prev,
          conversations: {
            ...prev.conversations,
            [conv.id]: updateMessage(conv, messageId, {
              rawContent: fullContent,
              content: [...contentBeforeTools, ...toolCallContent],
            }),
          },
        };
      });

      // Execute tool calls and create tool messages
      const toolResults: Array<{ toolCallId: string; result: string }> = [];
      const toolMessages: Array<ReturnType<typeof createToolMessage>> = [];
      let lastMessageId = messageId;

      for (const tc of toolCalls) {
        const toolCall: ToolCall = { id: tc.id, name: tc.name, arguments: tc.arguments };
        const result = executeToolCall(toolCall);
        toolResults.push(result);

        // Create a tool message for this result (chained to previous message)
        const toolMessage = createToolMessage(tc.id, result.result, lastMessageId);
        toolMessages.push(toolMessage);

        // Update lastMessageId for next iteration
        lastMessageId = toolMessage.id;
      }

      // Create a new assistant message for the continuation response (chained to last tool message)
      const continuationMessage = createMessage('assistant', '', lastMessageId);

      // Add all tool messages and continuation message in a single setState
      setState((prev) => {
        const conv = prev.conversations[conversation.id];
        if (!conv) return prev;

        let updatedConv = conv;

        // Add each tool message
        for (const toolMessage of toolMessages) {
          updatedConv = addMessage(updatedConv, toolMessage);
        }

        // Add continuation message
        updatedConv = addMessage(updatedConv, continuationMessage);

        return {
          ...prev,
          conversations: {
            ...prev.conversations,
            [conv.id]: updatedConv,
          },
        };
      });

      // Build messages with tool call and results for continuation
      const messagesWithTools: ApiMessage[] = [
        ...apiMessages,
        {
          role: 'assistant',
          content: fullContent || null,
          tool_calls: toolCalls.map((tc) => ({
            id: tc.id,
            type: 'function',
            function: { name: tc.name, arguments: tc.arguments },
          })),
        },
        ...toolResults.map((tr) => ({
          role: 'tool',
          content: tr.result,
          tool_call_id: tr.toolCallId,
        })),
      ];

      // Reset streaming state for continuation
      accumulatedToolCalls.clear();
      hasToolCalls = false;
      fullContent = '';

      // Continue the conversation with tool results, streaming into the new assistant message
      await streamToMessage(messagesWithTools, continuationMessage.id);
    }
  };

  const handleEditMessage = useCallback(
    (messageId: string, newContent: string, regenerate: boolean) => {
      if (!activeConversation) return;

      const originalMessage = activeConversation.messages[messageId];
      if (!originalMessage) return;

      if (regenerate && originalMessage.role === 'assistant') {
        // Create a new branch with edited content
        const newMessage = createMessage(
          'assistant',
          newContent,
          originalMessage.parentId
        );

        let updatedConversation = {
          ...activeConversation,
          messages: {
            ...activeConversation.messages,
            [newMessage.id]: newMessage,
          },
          activeLeafId: newMessage.id,
          updatedAt: Date.now(),
        };

        // Update parent's childIds
        if (originalMessage.parentId) {
          const parent = activeConversation.messages[originalMessage.parentId];
          updatedConversation.messages[originalMessage.parentId] = {
            ...parent,
            childIds: [...parent.childIds, newMessage.id],
          };
        }

        handleUpdateConversation(updatedConversation);

        // Regenerate from this point
        setIsStreaming(true);
        streamResponse(updatedConversation, newMessage.id, newContent)
          .catch(console.error)
          .finally(() => setIsStreaming(false));
      } else if (regenerate && originalMessage.role === 'tool') {
        // Create a new tool message branch with edited content
        const newToolMessage = createToolMessage(
          originalMessage.toolCallId || '',
          newContent,
          originalMessage.parentId
        );

        // Create a new assistant message as child of the tool message
        const newAssistantMessage = createMessage(
          'assistant',
          '',
          newToolMessage.id
        );

        let updatedConversation = {
          ...activeConversation,
          messages: {
            ...activeConversation.messages,
            [newToolMessage.id]: newToolMessage,
            [newAssistantMessage.id]: newAssistantMessage,
          },
          activeLeafId: newAssistantMessage.id,
          updatedAt: Date.now(),
        };

        // Update parent's childIds (add new tool message as sibling)
        if (originalMessage.parentId) {
          const parent = activeConversation.messages[originalMessage.parentId];
          updatedConversation.messages[originalMessage.parentId] = {
            ...parent,
            childIds: [...parent.childIds, newToolMessage.id],
          };
        }

        // Set the tool message's childIds to include the assistant message
        updatedConversation.messages[newToolMessage.id] = {
          ...newToolMessage,
          childIds: [newAssistantMessage.id],
        };

        handleUpdateConversation(updatedConversation);

        // Stream response to the new assistant message
        setIsStreaming(true);
        streamResponse(updatedConversation, newAssistantMessage.id)
          .catch(console.error)
          .finally(() => setIsStreaming(false));
      } else {
        // Just update the message content (for user messages or non-regenerate edits)
        const updatedConversation = updateMessage(
          activeConversation,
          messageId,
          {
            rawContent: newContent,
            content: [{ type: 'text', content: newContent }],
          }
        );
        handleUpdateConversation(updatedConversation);
      }
    },
    [activeConversation, handleUpdateConversation]
  );

  const handleSwitchBranch = useCallback(
    (parentId: string, childIndex: number) => {
      if (!activeConversation) return;

      const parent = activeConversation.messages[parentId];
      if (!parent || !parent.childIds[childIndex]) return;

      // Find the leaf of this branch
      let currentId = parent.childIds[childIndex];
      while (true) {
        const current = activeConversation.messages[currentId];
        if (!current || current.childIds.length === 0) break;
        currentId = current.childIds[0]; // Follow first child
      }

      handleUpdateConversation({
        ...activeConversation,
        activeLeafId: currentId,
      });
    },
    [activeConversation, handleUpdateConversation]
  );

  return (
    <div className="app">
      <Sidebar
        conversations={Object.values(state.conversations).sort(
          (a, b) => b.updatedAt - a.updatedAt
        )}
        activeId={state.activeConversationId}
        onNewChat={handleNewChat}
        onSelect={handleSelectConversation}
        onDelete={handleDeleteConversation}
      />

      <main className="main-content">
        {activeConversation ? (
          <ChatView
            conversation={activeConversation}
            isStreaming={isStreaming}
            onSendMessage={handleSendMessage}
            onEditMessage={handleEditMessage}
            onSwitchBranch={handleSwitchBranch}
            onShowSettings={() => setShowSettings(true)}
            onShowRaw={() => setShowRawPanel(true)}
          />
        ) : (
          <div className="empty-state">
            <h2>Welcome to ChatGPT Clone</h2>
            <p>Start a new conversation or select one from the sidebar.</p>
            <button className="new-chat-btn" onClick={handleNewChat}>
              + New Chat
            </button>
          </div>
        )}
      </main>

      {showSettings && activeConversation && (
        <SettingsPanel
          settings={activeConversation.settings}
          onUpdate={(settings) =>
            handleUpdateConversation({
              ...activeConversation,
              settings,
            })
          }
          onClose={() => setShowSettings(false)}
        />
      )}

      {showRawPanel && activeConversation && (
        <RawPanel
          conversation={activeConversation}
          onClose={() => setShowRawPanel(false)}
        />
      )}
    </div>
  );
}

export default App;
