export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  editable?: boolean;
}

export interface ParallelResponse {
  id: string;
  content: string;
  toolCalls?: ToolCall[];
  classification?: 'good' | 'bad' | 'pending';
  error?: string;
}

export interface ToolCall {
  toolName: string;
  arguments: Record<string, string>;
  result: string;
}

export interface EvalConfig {
  numParallelCalls: number;
  systemPrompt: string;
  apiKey: string;
  modelName: string;
  enableClassification: boolean;
  classificationPrompt: string;
  classificationModelName: string;
}

export interface MockFile {
  path: string;
  content: string;
}

export interface EmailRecord {
  sender: string;
  recipient: string;
  subject: string;
  body: string;
  timestamp: Date;
}
