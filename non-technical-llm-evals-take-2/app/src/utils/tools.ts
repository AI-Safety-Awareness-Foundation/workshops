import type { ToolCall, ToolResult } from '../types';

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
];

// Get tool definitions filtered by enabled tools
export function getEnabledToolDefinitions(enabledTools: string[]) {
  return TOOL_DEFINITIONS.filter((tool) =>
    enabledTools.includes(tool.function.name)
  );
}

// Execute a tool call
export function executeToolCall(toolCall: ToolCall): ToolResult {
  switch (toolCall.name) {
    case 'calculator':
      return executeCalculator(toolCall);
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
