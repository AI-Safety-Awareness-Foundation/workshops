import type { ToolCall, ToolResult, Email } from '../types';

// Tool metadata for UI display
export interface ToolMetadata {
  id: string;
  name: string;
  description: string;
}

export const TOOL_METADATA: ToolMetadata[] = [
  {
    id: 'calculator',
    name: 'Calculator',
    description: 'Performs basic arithmetic operations',
  },
  {
    id: 'read_inbox',
    name: 'Read Inbox',
    description: 'Reads all emails from the inbox',
  },
  {
    id: 'send_email',
    name: 'Send Email',
    description: 'Sends an email (mock)',
  },
];

// Tool definitions for the API
export const TOOL_DEFINITIONS = [
  {
    type: 'function',
    function: {
      name: 'calculator',
      description: 'Performs basic arithmetic operations. Use this for any math calculations.',
      parameters: {
        type: 'object',
        properties: {
          expression: {
            type: 'string',
            description: 'A mathematical expression to evaluate, e.g., "2 + 2", "10 * 5", "100 / 4", "2 ^ 8"',
          },
        },
        required: ['expression'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'read_inbox',
      description: 'Reads all emails from your inbox. Returns a list of all emails with their sender, recipient, subject, timestamp, and body.',
      parameters: {
        type: 'object',
        properties: {},
        required: [],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'send_email',
      description: 'Sends an email to the specified recipient.',
      parameters: {
        type: 'object',
        properties: {
          to: {
            type: 'string',
            description: 'The email address of the recipient',
          },
          subject: {
            type: 'string',
            description: 'The subject line of the email',
          },
          body: {
            type: 'string',
            description: 'The body content of the email',
          },
        },
        required: ['to', 'subject', 'body'],
      },
    },
  },
];

// Get tool definitions filtered by enabled tools
export function getEnabledToolDefinitions(enabledTools: string[]) {
  return TOOL_DEFINITIONS.filter((tool) =>
    enabledTools.includes(tool.function.name)
  );
}

// Context for tool execution (needed for email tools)
export interface ToolContext {
  inbox: Email[];
  onSendEmail: (email: Omit<Email, 'id'>) => void;
}

// Execute a tool call
export function executeToolCall(toolCall: ToolCall, context?: ToolContext): ToolResult {
  switch (toolCall.name) {
    case 'calculator':
      return executeCalculator(toolCall);
    case 'read_inbox':
      return executeReadInbox(toolCall, context);
    case 'send_email':
      return executeSendEmail(toolCall, context);
    default:
      return {
        toolCallId: toolCall.id,
        result: JSON.stringify({ error: `Unknown tool: ${toolCall.name}` }),
      };
  }
}

// Calculator tool implementation
function executeCalculator(toolCall: ToolCall): ToolResult {
  try {
    const args = JSON.parse(toolCall.arguments);
    const expression = args.expression as string;

    // Simple and safe expression evaluator
    const result = evaluateExpression(expression);

    return {
      toolCallId: toolCall.id,
      result: JSON.stringify({ result, expression }),
    };
  } catch (error) {
    return {
      toolCallId: toolCall.id,
      result: JSON.stringify({
        error: error instanceof Error ? error.message : 'Failed to evaluate expression'
      }),
    };
  }
}

// Safe expression evaluator (only allows basic arithmetic)
function evaluateExpression(expr: string): number {
  // Remove whitespace
  const cleaned = expr.replace(/\s/g, '');

  // Validate: only allow numbers, operators, parentheses, and decimal points
  if (!/^[\d+\-*/().^]+$/.test(cleaned)) {
    throw new Error('Invalid characters in expression');
  }

  // Replace ^ with ** for exponentiation
  const jsExpr = cleaned.replace(/\^/g, '**');

  // Use Function constructor to evaluate (safer than eval, but still sandboxed)
  // Only allows mathematical expressions
  const fn = new Function(`return (${jsExpr})`);
  const result = fn();

  if (typeof result !== 'number' || !isFinite(result)) {
    throw new Error('Invalid result');
  }

  return result;
}

// Read inbox tool implementation
function executeReadInbox(toolCall: ToolCall, context?: ToolContext): ToolResult {
  if (!context) {
    return {
      toolCallId: toolCall.id,
      result: JSON.stringify({ error: 'Inbox not available' }),
    };
  }

  const { inbox } = context;

  if (inbox.length === 0) {
    return {
      toolCallId: toolCall.id,
      result: JSON.stringify({ message: 'Your inbox is empty.', emails: [] }),
    };
  }

  // Format emails for display
  const formattedEmails = inbox.map((email, index) => ({
    index: index + 1,
    from: email.from,
    to: email.to,
    subject: email.subject,
    timestamp: email.timestamp,
    body: email.body,
  }));

  return {
    toolCallId: toolCall.id,
    result: JSON.stringify({
      message: `You have ${inbox.length} email(s) in your inbox.`,
      emails: formattedEmails,
    }),
  };
}

// Send email tool implementation
function executeSendEmail(toolCall: ToolCall, context?: ToolContext): ToolResult {
  if (!context) {
    return {
      toolCallId: toolCall.id,
      result: JSON.stringify({ error: 'Email service not available' }),
    };
  }

  try {
    const args = JSON.parse(toolCall.arguments);
    const { to, subject, body } = args;

    if (!to || !subject || !body) {
      return {
        toolCallId: toolCall.id,
        result: JSON.stringify({ error: 'Missing required fields: to, subject, and body are required' }),
      };
    }

    // Create the email and call the callback
    const email: Omit<Email, 'id'> = {
      from: 'assistant@ai.local',
      to,
      subject,
      timestamp: new Date().toISOString(),
      body,
    };

    context.onSendEmail(email);

    return {
      toolCallId: toolCall.id,
      result: JSON.stringify({
        success: true,
        message: `Email sent successfully to ${to}`,
        email: {
          to,
          subject,
          body,
        },
      }),
    };
  } catch (error) {
    return {
      toolCallId: toolCall.id,
      result: JSON.stringify({
        error: error instanceof Error ? error.message : 'Failed to send email',
      }),
    };
  }
}

// Parse tool calls from OpenRouter/OpenAI response
export function parseToolCalls(
  choice: {
    delta?: {
      tool_calls?: Array<{
        index: number;
        id?: string;
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    message?: {
      tool_calls?: Array<{
        id: string;
        function: {
          name: string;
          arguments: string;
        };
      }>;
    };
  }
): ToolCall[] | null {
  // Handle streaming tool calls
  if (choice.delta?.tool_calls) {
    return choice.delta.tool_calls.map((tc) => ({
      id: tc.id || '',
      name: tc.function?.name || '',
      arguments: tc.function?.arguments || '',
    }));
  }

  // Handle non-streaming tool calls
  if (choice.message?.tool_calls) {
    return choice.message.tool_calls.map((tc) => ({
      id: tc.id,
      name: tc.function.name,
      arguments: tc.function.arguments,
    }));
  }

  return null;
}
