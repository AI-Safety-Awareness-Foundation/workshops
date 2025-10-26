import React from 'react';
import { ParallelResponse } from '../types';

interface ResponseGridProps {
  responses: ParallelResponse[];
  isLoading: boolean;
  progress?: { completed: number; total: number };
}

export const ResponseGrid: React.FC<ResponseGridProps> = ({ responses, isLoading, progress }) => {
  if (responses.length === 0 && !isLoading) {
    return (
      <div className="response-grid-empty">
        <p>No responses yet. Configure your settings and send a message to start evaluation.</p>
      </div>
    );
  }

  return (
    <div className="response-grid-container">
      <div className="response-grid-header">
        <h3>Parallel Responses</h3>
        {isLoading && progress && (
          <div className="progress-indicator">
            Loading: {progress.completed} / {progress.total}
          </div>
        )}
      </div>

      <div className="response-grid">
        {responses.map((response) => (
          <div
            key={response.id}
            className={`response-card ${response.classification || ''} ${response.error ? 'error' : ''}`}
          >
            {response.error ? (
              <div className="response-error">
                <strong>Error:</strong> {response.error}
              </div>
            ) : (
              <>
                <div className="response-content">
                  {response.content}
                </div>

                {response.toolCalls && response.toolCalls.length > 0 && (
                  <div className="tool-calls">
                    <strong>Tool Calls:</strong>
                    {response.toolCalls.map((call, idx) => (
                      <div key={idx} className="tool-call">
                        <div className="tool-name">{call.toolName}</div>
                        <div className="tool-args">
                          {Object.entries(call.arguments).map(([key, value]) => (
                            <div key={key}>
                              <code>{key}: {value}</code>
                            </div>
                          ))}
                        </div>
                        <div className="tool-result">
                          Result: <code>{call.result}</code>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
