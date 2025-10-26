import Anthropic from '@anthropic-ai/sdk';
import { Message, ParallelResponse, ToolCall } from '../types';
import { MockFilesystem } from '../mockTools/filesystem';
import { MockEmailService } from '../mockTools/email';

export class LLMService {
  private client: Anthropic | null = null;
  private filesystem: MockFilesystem;
  private emailService: MockEmailService;

  constructor() {
    this.filesystem = new MockFilesystem();
    this.emailService = new MockEmailService();
  }

  setApiKey(apiKey: string) {
    this.client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
  }

  private getTools() {
    return [
      {
        name: 'read_file',
        description: 'Read a file from the mock filesystem',
        input_schema: {
          type: 'object' as const,
          properties: {
            path: {
              type: 'string' as const,
              description: 'The path to the file to read'
            }
          },
          required: ['path']
        }
      },
      {
        name: 'write_file',
        description: 'Write content to a file in the mock filesystem',
        input_schema: {
          type: 'object' as const,
          properties: {
            path: {
              type: 'string' as const,
              description: 'The path to the file to write'
            },
            content: {
              type: 'string' as const,
              description: 'The content to write to the file'
            }
          },
          required: ['path', 'content']
        }
      },
      {
        name: 'list_directory',
        description: 'List files in a directory',
        input_schema: {
          type: 'object' as const,
          properties: {
            path: {
              type: 'string' as const,
              description: 'The directory path to list'
            }
          },
          required: ['path']
        }
      },
      {
        name: 'emailer',
        description: 'Send an email',
        input_schema: {
          type: 'object' as const,
          properties: {
            sender_address: {
              type: 'string' as const,
              description: 'The sender email address'
            },
            recipient_address: {
              type: 'string' as const,
              description: 'The recipient email address'
            },
            subject: {
              type: 'string' as const,
              description: 'The email subject'
            },
            body: {
              type: 'string' as const,
              description: 'The email body'
            }
          },
          required: ['sender_address', 'recipient_address', 'subject', 'body']
        }
      }
    ];
  }

  private executeTool(toolName: string, args: Record<string, string>): string {
    try {
      switch (toolName) {
        case 'read_file':
          return this.filesystem.read(args.path);
        case 'write_file':
          this.filesystem.write(args.path, args.content);
          return `File written successfully to ${args.path}`;
        case 'list_directory':
          const files = this.filesystem.list(args.path);
          return files.length > 0 ? files.join('\n') : 'Directory is empty';
        case 'emailer':
          return this.emailService.send(
            args.sender_address,
            args.recipient_address,
            args.subject,
            args.body
          );
        default:
          return `Unknown tool: ${toolName}`;
      }
    } catch (error) {
      return `Error: ${error instanceof Error ? error.message : String(error)}`;
    }
  }

  async callModel(
    messages: Message[],
    systemPrompt: string,
    modelName: string = 'claude-3-5-sonnet-20241022'
  ): Promise<{ content: string; toolCalls: ToolCall[] }> {
    if (!this.client) {
      throw new Error('API key not set');
    }

    const anthropicMessages = messages
      .filter(m => m.role !== 'system')
      .map(m => ({
        role: m.role as 'user' | 'assistant',
        content: m.content
      }));

    const toolCalls: ToolCall[] = [];
    let response = await this.client.messages.create({
      model: modelName,
      max_tokens: 4096,
      system: systemPrompt,
      messages: anthropicMessages,
      tools: this.getTools()
    });

    // Handle tool use
    while (response.stop_reason === 'tool_use') {
      const toolUseBlock = response.content.find(block => block.type === 'tool_use');
      if (!toolUseBlock || toolUseBlock.type !== 'tool_use') break;

      const toolResult = this.executeTool(toolUseBlock.name, toolUseBlock.input as Record<string, string>);

      toolCalls.push({
        toolName: toolUseBlock.name,
        arguments: toolUseBlock.input as Record<string, string>,
        result: toolResult
      });

      // Continue conversation with tool result
      response = await this.client.messages.create({
        model: modelName,
        max_tokens: 4096,
        system: systemPrompt,
        messages: [
          ...anthropicMessages,
          {
            role: 'assistant',
            content: response.content
          },
          {
            role: 'user',
            content: [{
              type: 'tool_result',
              tool_use_id: toolUseBlock.id,
              content: toolResult
            }]
          }
        ],
        tools: this.getTools()
      });
    }

    const textContent = response.content
      .filter(block => block.type === 'text')
      .map(block => block.type === 'text' ? block.text : '')
      .join('');

    return {
      content: textContent,
      toolCalls
    };
  }

  async runParallelCalls(
    messages: Message[],
    systemPrompt: string,
    numCalls: number,
    modelName: string,
    onProgress?: (completed: number, total: number) => void
  ): Promise<ParallelResponse[]> {
    const promises = Array(numCalls).fill(null).map(async (_, index) => {
      try {
        const { content, toolCalls } = await this.callModel(messages, systemPrompt, modelName);
        if (onProgress) onProgress(index + 1, numCalls);
        return {
          id: `response-${index}`,
          content,
          toolCalls,
          classification: 'pending' as const
        };
      } catch (error) {
        if (onProgress) onProgress(index + 1, numCalls);
        return {
          id: `response-${index}`,
          content: '',
          error: error instanceof Error ? error.message : String(error),
          classification: 'pending' as const
        };
      }
    });

    return Promise.all(promises);
  }

  getFilesystem(): MockFilesystem {
    return this.filesystem;
  }

  getEmailService(): MockEmailService {
    return this.emailService;
  }
}
