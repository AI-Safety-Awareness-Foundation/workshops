import { useState } from 'react';
import type { Conversation } from '../types';
import { conversationToPlainText, conversationToJSON } from '../utils/helpers';

interface RawPanelProps {
  conversation: Conversation;
  onClose: () => void;
}

type TabType = 'plain' | 'json';

function RawPanel({ conversation, onClose }: RawPanelProps) {
  const [activeTab, setActiveTab] = useState<TabType>('plain');

  const content =
    activeTab === 'plain'
      ? conversationToPlainText(conversation)
      : conversationToJSON(conversation);

  return (
    <div className="raw-panel">
      <div className="raw-panel-header">
        <span className="raw-panel-title">Raw Conversation</span>
        <button className="close-btn" onClick={onClose}>
          x
        </button>
      </div>

      <div className="raw-panel-tabs">
        <button
          className={`raw-panel-tab ${activeTab === 'plain' ? 'active' : ''}`}
          onClick={() => setActiveTab('plain')}
        >
          Plain Text
        </button>
        <button
          className={`raw-panel-tab ${activeTab === 'json' ? 'active' : ''}`}
          onClick={() => setActiveTab('json')}
        >
          JSON
        </button>
      </div>

      <div className="raw-panel-content">
        <pre className="raw-content">{content}</pre>
      </div>
    </div>
  );
}

export default RawPanel;
