import { useState, useRef } from 'react';
import { ConfigPanel } from './components/ConfigPanel';
import { MessageEditor } from './components/MessageEditor';
import { ResponseGrid } from './components/ResponseGrid';
import { ChatInput } from './components/ChatInput';
import { LLMService } from './services/llmService';
import { EvalConfig, Message, ParallelResponse } from './types';
import './App.css';

function App() {
  const [config, setConfig] = useState<EvalConfig>({
    numParallelCalls: 5,
    systemPrompt: 'You are a helpful AI assistant.',
    apiKey: '',
    modelName: 'claude-3-5-sonnet-20241022',
    enableClassification: false
  });

  const [prefixMessages, setPrefixMessages] = useState<Message[]>([]);
  const [responses, setResponses] = useState<ParallelResponse[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState<{ completed: number; total: number } | undefined>();

  const llmServiceRef = useRef(new LLMService());

  const handleSendMessage = async (messageContent: string) => {
    if (!config.apiKey) {
      alert('Please enter your API key first');
      return;
    }

    const llmService = llmServiceRef.current;
    llmService.setApiKey(config.apiKey);

    const newMessage: Message = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content: messageContent
    };

    const conversationMessages = [...prefixMessages, newMessage];

    setIsLoading(true);
    setResponses([]);
    setProgress({ completed: 0, total: config.numParallelCalls });

    try {
      const parallelResponses = await llmService.runParallelCalls(
        conversationMessages,
        config.systemPrompt,
        config.numParallelCalls,
        config.modelName,
        (completed, total) => {
          setProgress({ completed, total });
        }
      );

      setResponses(parallelResponses);
    } catch (error) {
      alert(`Error: ${error instanceof Error ? error.message : String(error)}`);
    } finally {
      setIsLoading(false);
      setProgress(undefined);
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>LLM Manual Evaluation Interface</h1>
        <p>Evaluate LLM behavior through parallel response sampling</p>
      </header>

      <div className="app-layout">
        <aside className="sidebar">
          <ConfigPanel config={config} onConfigChange={setConfig} />
          <MessageEditor messages={prefixMessages} onMessagesChange={setPrefixMessages} />
        </aside>

        <main className="main-content">
          <div className="evaluation-area">
            <ChatInput onSend={handleSendMessage} disabled={isLoading} />
            <ResponseGrid responses={responses} isLoading={isLoading} progress={progress} />
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
