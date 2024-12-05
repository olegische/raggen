import { PromptService, PromptMessage } from '../prompt.service';
import { Message } from '@prisma/client';
import { ContextSearchResult } from '../context.service';
import { SYSTEM_PROMPTS } from '../../config/prompts';

describe('PromptService', () => {
  let promptService: PromptService;

  const mockMessage: Message = {
    id: 1,
    chatId: 'chat-1',
    message: 'Test message',
    model: 'test-model',
    provider: 'test-provider',
    temperature: 0.7,
    maxTokens: 1000,
    timestamp: new Date(),
    response: null
  };

  beforeEach(() => {
    promptService = new PromptService();
  });

  describe('formatPromptWithContext', () => {
    const mockContext: ContextSearchResult[] = [
      {
        message: { ...mockMessage, message: 'Context message 1' },
        score: 0.9,
        usedInPrompt: false
      },
      {
        message: { ...mockMessage, message: 'Context message 2' },
        score: 0.8,
        usedInPrompt: false
      },
      {
        message: { ...mockMessage, message: 'Context message 3' },
        score: 0.6,
        usedInPrompt: false
      }
    ];

    it('should format prompt with context', () => {
      const message = 'Test question';
      const result = promptService.formatPromptWithContext(message, mockContext, 'yandex');

      expect(result).toHaveLength(2);
      expect(result[0].role).toBe('system');
      expect(result[0].content).toBe(SYSTEM_PROMPTS.yandex);
      expect(result[1].role).toBe('user');
      expect(result[1].content).toContain('Test question');
      expect(result[1].content).toContain('Context message 1');
      expect(result[1].content).toContain('Context message 2');
      expect(result[1].content).not.toContain('Context message 3'); // Score too low
    });

    it('should respect maxContextMessages option', () => {
      const message = 'Test question';
      const result = promptService.formatPromptWithContext(message, mockContext, 'yandex', {
        maxContextMessages: 1
      });

      expect(result[1].content).toContain('Context message 1');
      expect(result[1].content).not.toContain('Context message 2');
    });

    it('should respect contextScoreThreshold option', () => {
      const message = 'Test question';
      const result = promptService.formatPromptWithContext(message, mockContext, 'yandex', {
        contextScoreThreshold: 0.85
      });

      expect(result[1].content).toContain('Context message 1');
      expect(result[1].content).not.toContain('Context message 2');
    });

    it('should mark used context messages', () => {
      const message = 'Test question';
      const context = [...mockContext];
      promptService.formatPromptWithContext(message, context, 'yandex');

      expect(context[0].usedInPrompt).toBe(true);
      expect(context[1].usedInPrompt).toBe(true);
      expect(context[2].usedInPrompt).toBe(false);
    });

    it('should format prompt without context if none provided', () => {
      const message = 'Test question';
      const result = promptService.formatPromptWithContext(message, [], 'yandex');

      expect(result[1].content).toBe(message);
    });
  });

  describe('formatMessageHistory', () => {
    const mockMessages: Message[] = [
      { ...mockMessage, provider: 'system', message: 'System message' },
      { ...mockMessage, provider: 'user', message: 'User message' },
      { ...mockMessage, provider: 'assistant', message: 'Assistant message' }
    ];

    it('should format message history correctly', () => {
      const result = promptService.formatMessageHistory(mockMessages);

      expect(result).toHaveLength(3);
      expect(result[0]).toEqual({ role: 'system', content: 'System message' });
      expect(result[1]).toEqual({ role: 'user', content: 'User message' });
      expect(result[2]).toEqual({ role: 'assistant', content: 'Assistant message' });
    });

    it('should handle empty message history', () => {
      const result = promptService.formatMessageHistory([]);
      expect(result).toHaveLength(0);
    });
  });
}); 