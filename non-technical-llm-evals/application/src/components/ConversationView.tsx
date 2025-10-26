import React, { useState } from 'react';
import { Message } from '../types';

interface ConversationViewProps {
  messages: Message[];
  onMessagesChange: (messages: Message[]) => void;
  onEvaluate: (finalMessage: string) => void;
  isLoading: boolean;
}

export const ConversationView: React.FC<ConversationViewProps> = ({
  messages,
  onMessagesChange,
  onEvaluate,
  isLoading
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [evaluationInput, setEvaluationInput] = useState('');

  const addMessage = (role: 'user' | 'assistant') => {
    const newMessage: Message = {
      id: `msg-${Date.now()}`,
      role,
      content: '',
      editable: true
    };
    onMessagesChange([...messages, newMessage]);
    setEditingId(newMessage.id);
  };

  const updateMessage = (id: string, content: string) => {
    onMessagesChange(messages.map(msg =>
      msg.id === id ? { ...msg, content } : msg
    ));
  };

  const deleteMessage = (id: string) => {
    onMessagesChange(messages.filter(msg => msg.id !== id));
  };

  const handleEvaluate = (e: React.FormEvent) => {
    e.preventDefault();
    if (evaluationInput.trim() && !isLoading) {
      onEvaluate(evaluationInput);
      setEvaluationInput('');
    }
  };

  return (
    <div className="conversation-view">
      <div className="conversation-header">
        <h3>Conversation Setup</h3>
        <p className="help-text">
          Build a conversation that the model will continue from, then add your evaluation message below.
        </p>
      </div>

      <div className="conversation-thread">
        {messages.length === 0 ? (
          <div className="conversation-empty">
            <p>No conversation prefix yet. Add messages below or start directly with your evaluation prompt.</p>
          </div>
        ) : (
          messages.map((msg) => (
            <div key={msg.id} className={`conversation-message conversation-message-${msg.role}`}>
              <div className="conversation-message-header">
                <span className="role-badge">{msg.role.toUpperCase()}</span>
                <button
                  onClick={() => deleteMessage(msg.id)}
                  className="delete-btn"
                  disabled={isLoading}
                >
                  Delete
                </button>
              </div>
              {editingId === msg.id ? (
                <textarea
                  value={msg.content}
                  onChange={(e) => updateMessage(msg.id, e.target.value)}
                  onBlur={() => setEditingId(null)}
                  autoFocus
                  rows={3}
                  className="conversation-message-edit"
                />
              ) : (
                <div
                  className="conversation-message-content"
                  onClick={() => !isLoading && setEditingId(msg.id)}
                >
                  {msg.content || <em>Click to edit...</em>}
                </div>
              )}
            </div>
          ))
        )}
      </div>

      <div className="conversation-actions">
        <button
          onClick={() => addMessage('user')}
          className="add-message-btn"
          disabled={isLoading}
        >
          + Add User Message
        </button>
        <button
          onClick={() => addMessage('assistant')}
          className="add-message-btn"
          disabled={isLoading}
        >
          + Add Assistant Message
        </button>
      </div>

      <div className="evaluation-input-section">
        <div className="evaluation-input-header">
          <h3>Evaluation Prompt</h3>
          <p className="help-text">
            This message will be added to the conversation and evaluated {messages.length > 0 ? 'as a continuation' : ''}.
          </p>
        </div>
        <form onSubmit={handleEvaluate} className="evaluation-form">
          <textarea
            value={evaluationInput}
            onChange={(e) => setEvaluationInput(e.target.value)}
            placeholder="Enter your evaluation prompt here..."
            disabled={isLoading}
            rows={4}
            className="evaluation-textarea"
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleEvaluate(e);
              }
            }}
          />
          <button
            type="submit"
            disabled={isLoading || !evaluationInput.trim()}
            className="evaluate-btn"
          >
            {isLoading ? 'Running Evaluation...' : 'Run Evaluation'}
          </button>
        </form>
      </div>
    </div>
  );
};
