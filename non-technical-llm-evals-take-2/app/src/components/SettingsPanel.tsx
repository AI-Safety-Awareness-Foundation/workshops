import type { ConversationSettings, ThinkingTokenFormat, EndpointType } from '../types';
import { COMMON_MODELS } from '../types';

interface SettingsPanelProps {
  settings: ConversationSettings;
  onUpdate: (settings: ConversationSettings) => void;
  onClose: () => void;
}

function SettingsPanel({ settings, onUpdate, onClose }: SettingsPanelProps) {
  const handleChange = <K extends keyof ConversationSettings>(
    key: K,
    value: ConversationSettings[K]
  ) => {
    onUpdate({ ...settings, [key]: value });
  };

  const isCustomModel = !COMMON_MODELS.some((m) => m.id === settings.model);

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <span className="settings-title">Settings</span>
        <button className="close-btn" onClick={onClose}>
          x
        </button>
      </div>

      <div className="settings-content">
        <div className="settings-section">
          <label className="settings-label">Endpoint Type</label>
          <select
            className="settings-select"
            value={settings.endpointType}
            onChange={(e) =>
              handleChange('endpointType', e.target.value as EndpointType)
            }
          >
            <option value="openrouter">OpenRouter</option>
            <option value="vllm">Custom VLLM</option>
          </select>
        </div>

        <div className="settings-section">
          <label className="settings-label">API Key</label>
          <input
            type="password"
            className="settings-input"
            value={settings.apiKey}
            onChange={(e) => handleChange('apiKey', e.target.value)}
            placeholder="Enter your API key"
          />
        </div>

        {settings.endpointType === 'vllm' && (
          <div className="settings-section">
            <label className="settings-label">VLLM URL</label>
            <input
              type="text"
              className="settings-input"
              value={settings.vllmUrl}
              onChange={(e) => handleChange('vllmUrl', e.target.value)}
              placeholder="http://localhost:8000"
            />
          </div>
        )}

        {settings.endpointType === 'openrouter' && (
          <div className="settings-section">
            <label className="settings-label">Model</label>
            <select
              className="settings-select"
              value={isCustomModel ? 'custom' : settings.model}
              onChange={(e) => {
                if (e.target.value !== 'custom') {
                  handleChange('model', e.target.value);
                }
              }}
            >
              {COMMON_MODELS.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name}
                </option>
              ))}
              <option value="custom">Custom...</option>
            </select>
            {isCustomModel && (
              <input
                type="text"
                className="settings-input"
                value={settings.model}
                onChange={(e) => handleChange('model', e.target.value)}
                placeholder="Enter model ID"
                style={{ marginTop: '8px' }}
              />
            )}
          </div>
        )}

        <div className="settings-section">
          <label className="settings-label">Thinking Token Format</label>
          <select
            className="settings-select"
            value={settings.thinkingTokenFormat}
            onChange={(e) =>
              handleChange(
                'thinkingTokenFormat',
                e.target.value as ThinkingTokenFormat
              )
            }
          >
            <option value="inline-tags">
              Inline Tags (&lt;think&gt;...&lt;/think&gt;)
            </option>
            <option value="separate-field">Separate Field in Response</option>
          </select>
        </div>

        <div className="settings-section">
          <label className="settings-label">System Prompt</label>
          <textarea
            className="settings-textarea"
            value={settings.systemPrompt}
            onChange={(e) => handleChange('systemPrompt', e.target.value)}
            placeholder="Enter system prompt..."
          />
        </div>
      </div>
    </div>
  );
}

export default SettingsPanel;
