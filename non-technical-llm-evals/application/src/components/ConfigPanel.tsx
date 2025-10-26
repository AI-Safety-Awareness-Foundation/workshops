import React from 'react';
import { EvalConfig } from '../types';

interface ConfigPanelProps {
  config: EvalConfig;
  onConfigChange: (config: EvalConfig) => void;
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({ config, onConfigChange }) => {
  return (
    <div className="config-panel">
      <h3>Configuration</h3>

      <div className="config-field">
        <label htmlFor="api-key">API Key:</label>
        <input
          id="api-key"
          type="password"
          value={config.apiKey}
          onChange={(e) => onConfigChange({ ...config, apiKey: e.target.value })}
          placeholder="Enter your Anthropic API key"
        />
      </div>

      <div className="config-field">
        <label htmlFor="model-name">Model:</label>
        <select
          id="model-name"
          value={config.modelName}
          onChange={(e) => onConfigChange({ ...config, modelName: e.target.value })}
        >
          <option value="claude-3-5-sonnet-20241022">Claude 3.5 Sonnet</option>
          <option value="claude-3-opus-20240229">Claude 3 Opus</option>
          <option value="claude-3-haiku-20240307">Claude 3 Haiku</option>
        </select>
      </div>

      <div className="config-field">
        <label htmlFor="num-calls">Number of Parallel Calls:</label>
        <input
          id="num-calls"
          type="number"
          min="1"
          max="50"
          value={config.numParallelCalls}
          onChange={(e) => onConfigChange({ ...config, numParallelCalls: parseInt(e.target.value) })}
        />
      </div>

      <div className="config-field">
        <label htmlFor="system-prompt">System Prompt:</label>
        <textarea
          id="system-prompt"
          value={config.systemPrompt}
          onChange={(e) => onConfigChange({ ...config, systemPrompt: e.target.value })}
          rows={6}
          placeholder="Enter system prompt..."
        />
      </div>

      <div className="config-field">
        <label>
          <input
            type="checkbox"
            checked={config.enableClassification}
            onChange={(e) => onConfigChange({ ...config, enableClassification: e.target.checked })}
          />
          Enable automatic classification (coming soon)
        </label>
      </div>
    </div>
  );
};
