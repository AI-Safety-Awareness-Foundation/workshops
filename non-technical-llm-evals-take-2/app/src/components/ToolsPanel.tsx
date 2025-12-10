import type { ConversationSettings } from '../types';
import { TOOL_METADATA } from '../utils/tools';

interface ToolsPanelProps {
  settings: ConversationSettings;
  onUpdate: (settings: ConversationSettings) => void;
  onClose: () => void;
}

function ToolsPanel({ settings, onUpdate, onClose }: ToolsPanelProps) {
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
      </div>
    </div>
  );
}

export default ToolsPanel;
