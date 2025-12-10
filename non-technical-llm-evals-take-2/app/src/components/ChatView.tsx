import { useState, useRef, useEffect } from 'react';
import type { Conversation, Message } from '../types';
import { getActivePath } from '../utils/helpers';
import MessageComponent from './Message';

interface ChatViewProps {
  conversation: Conversation;
  isStreaming: boolean;
  onSendMessage: (content: string, prefill?: string) => void;
  onEditMessage: (messageId: string, newContent: string, regenerate: boolean) => void;
  onContinueMessage: (messageId: string, prefill: string) => void;
  onSwitchBranch: (parentId: string, childIndex: number) => void;
  onShowSettings: () => void;
  onShowRaw: () => void;
}

function ChatView({
  conversation,
  isStreaming,
  onSendMessage,
  onEditMessage,
  onContinueMessage,
  onSwitchBranch,
  onShowSettings,
  onShowRaw,
}: ChatViewProps) {
  const [input, setInput] = useState('');
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const activePath = getActivePath(conversation);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [activePath]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isStreaming) return;
    onSendMessage(input.trim());
    setInput('');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Find which branch we're on for each message with siblings
  const getBranchInfo = (message: Message) => {
    if (!message.parentId) return null;
    const parent = conversation.messages[message.parentId];
    if (!parent || parent.childIds.length <= 1) return null;

    const currentIndex = parent.childIds.indexOf(message.id);
    return {
      parentId: parent.id,
      current: currentIndex + 1,
      total: parent.childIds.length,
    };
  };

  return (
    <>
      <header className="chat-header">
        <span className="chat-header-title">{conversation.title}</span>
        <div className="chat-header-actions">
          <button className="header-btn" onClick={onShowRaw}>
            Raw View
          </button>
          <button className="header-btn" onClick={onShowSettings}>
            Settings
          </button>
        </div>
      </header>

      <div className="messages-container">
        <div className="messages-wrapper">
          {activePath.map((message) => {
            const branchInfo = getBranchInfo(message);
            return (
              <MessageComponent
                key={message.id}
                message={message}
                branchInfo={branchInfo}
                isEditing={editingMessageId === message.id}
                isStreaming={isStreaming && message.id === activePath[activePath.length - 1]?.id}
                onStartEdit={() => setEditingMessageId(message.id)}
                onCancelEdit={() => setEditingMessageId(null)}
                onSaveEdit={(content, regenerate) => {
                  onEditMessage(message.id, content, regenerate);
                  setEditingMessageId(null);
                }}
                onContinue={(prefill) => {
                  onContinueMessage(message.id, prefill);
                }}
                onSwitchBranch={(index) => {
                  if (branchInfo) {
                    onSwitchBranch(branchInfo.parentId, index);
                  }
                }}
              />
            );
          })}
          <div ref={messagesEndRef} />
        </div>
      </div>

      <div className="input-area">
        <div className="input-wrapper">
          <form onSubmit={handleSubmit}>
            <div className="input-container">
              <textarea
                ref={textareaRef}
                className="message-input"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a message..."
                disabled={isStreaming}
                rows={1}
              />
              <button
                type="submit"
                className="send-btn"
                disabled={!input.trim() || isStreaming}
              >
                {isStreaming ? 'Sending...' : 'Send'}
              </button>
            </div>
          </form>
        </div>
      </div>
    </>
  );
}

export default ChatView;
