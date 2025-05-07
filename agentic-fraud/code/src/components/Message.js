import React from 'react';
import './Message.css';

const Message = ({ message }) => {
  const { text, sender } = message;
  
  return (
    <div className={`message ${sender}`}>
      <div className="message-avatar">
        {sender === 'assistant' ? 'C' : 'U'}
      </div>
      <div className="message-content">
        <div className="message-text">{text}</div>
      </div>
    </div>
  );
};

export default Message;