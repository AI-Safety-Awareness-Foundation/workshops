import React, { useState } from 'react';
import { Message } from '../types';

interface MessageEditorProps {
  messages: Message[];
  onMessagesChange: (messages: Message[]) => void;
}

export const MessageEditor: React.FC<MessageEditorProps> = ({ messages, onMessagesChange }) => {
  const [editingId, setEditingId] = useState<string | null>(null);

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

  return (
    <div className="message-editor">
      <h3>Conversation Prefix</h3>
      <p className="help-text">
        Build a multi-turn conversation that the model will continue from.
      </p>

      <div className="messages-list">
        {messages.map((msg) => (
          <div key={msg.id} className={`message message-${msg.role}`}>
            <div className="message-header">
              <strong>{msg.role.toUpperCase()}</strong>
              <button onClick={() => deleteMessage(msg.id)} className="delete-btn">
                Delete
              </button>
            </div>
            {editingId === msg.id ? (
              <textarea
                value={msg.content}
                onChange={(e) => updateMessage(msg.id, e.target.value)}
                onBlur={() => setEditingId(null)}
                autoFocus
                rows={4}
              />
            ) : (
              <div
                className="message-content"
                onClick={() => setEditingId(msg.id)}
              >
                {msg.content || <em>Click to edit...</em>}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="add-message-buttons">
        <button onClick={() => addMessage('user')}>Add User Message</button>
        <button onClick={() => addMessage('assistant')}>Add Assistant Message</button>
      </div>
    </div>
  );
};
