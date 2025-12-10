import type { ConversationSettings } from '../types';
import { TOOL_METADATA } from '../utils/tools';

interface ToolsPanelProps {
  settings: ConversationSettings;
  onUpdate: (settings: ConversationSettings) => void;
  onClose: () => void;
  onShowInboxEditor: () => void;
}

function ToolsPanel({ settings, onUpdate, onClose, onShowInboxEditor }: ToolsPanelProps) {
  const handleToggleTool = (toolId: string) => {
    const isEnabled = settings.enabledTools.includes(toolId);
    const newEnabledTools = isEnabled
      ? settings.enabledTools.filter((id) => id !== toolId)
      : [...settings.enabledTools, toolId];
    onUpdate({ ...settings, enabledTools: newEnabledTools });
  };

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <span className="settings-title">Tools</span>
        <button className="close-btn" onClick={onClose}>
          x
        </button>
      </div>

      <div className="settings-content">
        <p style={{ color: 'var(--text-secondary)', marginBottom: '16px', fontSize: '14px' }}>
          Enable or disable tools available to the model for this conversation.
        </p>

        {TOOL_METADATA.map((tool) => {
          const isEnabled = settings.enabledTools.includes(tool.id);
          return (
            <div key={tool.id} className="settings-section">
              <label
                style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '12px',
                  cursor: 'pointer',
                }}
              >
                <input
                  type="checkbox"
                  checked={isEnabled}
                  onChange={() => handleToggleTool(tool.id)}
                  style={{
                    marginTop: '4px',
                    width: '16px',
                    height: '16px',
                    cursor: 'pointer',
                  }}
                />
                <div>
                  <div style={{ fontWeight: 500, marginBottom: '4px' }}>
                    {tool.name}
                  </div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '13px' }}>
                    {tool.description}
                  </div>
                </div>
              </label>
            </div>
          );
        })}

        <div style={{ borderTop: '1px solid var(--border-color)', marginTop: '16px', paddingTop: '16px' }}>
          <div style={{ marginBottom: '8px', fontWeight: 500 }}>Email Inbox</div>
          <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '12px' }}>
            Configure the mock emails for the read_inbox and send_email tools.
          </p>
          <button
            onClick={onShowInboxEditor}
            style={{
              padding: '8px 16px',
              backgroundColor: 'var(--bg-tertiary)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '4px',
              cursor: 'pointer',
              width: '100%',
            }}
          >
            Edit Inbox ({settings.inbox.length} emails)
          </button>
        </div>
      </div>
    </div>
  );
}

export default ToolsPanel;
