import { useState } from 'react';
import type { Message, MessageContent } from '../types';

interface BranchInfo {
  parentId: string;
  current: number;
  total: number;
}

interface MessageProps {
  message: Message;
  branchInfo: BranchInfo | null;
  isEditing: boolean;
  isStreaming: boolean;
  onStartEdit: () => void;
  onCancelEdit: () => void;
  onSaveEdit: (content: string, regenerate: boolean) => void;
  onContinue: (prefill: string) => void;
  onSwitchBranch: (index: number) => void;
}

function MessageComponent({
  message,
  branchInfo,
  isEditing,
  isStreaming,
  onStartEdit,
  onCancelEdit,
  onSaveEdit,
  onContinue,
  onSwitchBranch,
}: MessageProps) {
  const [editContent, setEditContent] = useState(message.rawContent);
  const [expandedThinking, setExpandedThinking] = useState<Set<number>>(new Set());
  const [expandedTools, setExpandedTools] = useState<Set<number>>(new Set());

  const toggleThinking = (index: number) => {
    setExpandedThinking((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const toggleTool = (index: number) => {
    setExpandedTools((prev) => {
      const next = new Set(prev);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return next;
    });
  };

  const renderContent = (content: MessageContent, index: number) => {
    switch (content.type) {
      case 'thinking':
        return (
          <div key={index} className="thinking-block">
            <div
              className="thinking-header"
              onClick={() => toggleThinking(index)}
            >
              <span>{expandedThinking.has(index) ? '▼' : '▶'}</span>
              <span>Thinking...</span>
            </div>
            {expandedThinking.has(index) && (
              <div className="thinking-content">{content.content}</div>
            )}
          </div>
        );

      case 'tool_call':
        return (
          <div key={index} className="tool-block">
            <div className="tool-header" onClick={() => toggleTool(index)}>
              <span>{expandedTools.has(index) ? '▼' : '▶'}</span>
              <span>Tool: {content.toolCall.name}</span>
            </div>
            {expandedTools.has(index) && (
              <div className="tool-content">
                <div className="tool-section">
                  <div className="tool-section-label">Arguments</div>
                  <div className="tool-section-content">
                    {content.toolCall.arguments}
                  </div>
                </div>
                {content.result && (
                  <div className="tool-section">
                    <div className="tool-section-label">Result</div>
                    <div className="tool-section-content">
                      {content.result.result}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        );

      case 'text':
      default:
        return (
          <div key={index} className="message-text">
            {content.content}
          </div>
        );
    }
  };

  if (isEditing) {
    return (
      <div className="message message-editing">
        <div className="message-header">
          <span className={`message-role ${message.role}`}>
            {message.role === 'user' ? 'You' : 'Assistant'}
          </span>
        </div>
        <textarea
          className="edit-textarea"
          value={editContent}
          onChange={(e) => setEditContent(e.target.value)}
          autoFocus
        />
        <div className="edit-actions">
          <button
            className="edit-btn save"
            onClick={() => onSaveEdit(editContent, false)}
          >
            Save
          </button>
          {message.role === 'assistant' && (
            <button
              className="edit-btn save"
              onClick={() => onSaveEdit(editContent, true)}
            >
              Save & Continue
            </button>
          )}
          <button className="edit-btn cancel" onClick={onCancelEdit}>
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="message">
      <div className="message-header">
        <span className={`message-role ${message.role}`}>
          {message.role === 'user' ? 'You' : 'Assistant'}
        </span>
        {isStreaming && <span className="streaming-indicator" />}
        <div className="message-actions">
          <button className="message-action-btn" onClick={onStartEdit}>
            Edit
          </button>
          {message.role === 'assistant' && (
            <button
              className="message-action-btn"
              onClick={() => onContinue(message.rawContent)}
            >
              Continue
            </button>
          )}
        </div>
      </div>

      {branchInfo && (
        <div className="branch-indicator">
          <button
            className="branch-nav-btn"
            onClick={() => onSwitchBranch(branchInfo.current - 2)}
            disabled={branchInfo.current <= 1}
          >
            {'<'}
          </button>
          <span>
            {branchInfo.current} / {branchInfo.total}
          </span>
          <button
            className="branch-nav-btn"
            onClick={() => onSwitchBranch(branchInfo.current)}
            disabled={branchInfo.current >= branchInfo.total}
          >
            {'>'}
          </button>
        </div>
      )}

      <div className="message-content">
        {message.content.map((content, index) => renderContent(content, index))}
      </div>
    </div>
  );
}

export default MessageComponent;
