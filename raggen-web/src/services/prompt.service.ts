import { Message } from '@prisma/client';
import { SYSTEM_PROMPTS, SystemPromptType } from '../config/prompts';
import { ContextSearchResult } from './context.service';

export interface PromptMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface PromptFormatOptions {
  maxContextLength?: number;
  maxContextMessages?: number;
  contextScoreThreshold?: number;
}

export class PromptService {
  private static readonly DEFAULT_MAX_CONTEXT_LENGTH = 2000;
  private static readonly DEFAULT_MAX_CONTEXT_MESSAGES = 3;
  private static readonly DEFAULT_CONTEXT_SCORE_THRESHOLD = 0.7;

  /**
   * Форматирование промпта с контекстом
   */
  formatPromptWithContext(
    message: string,
    context: ContextSearchResult[],
    provider: SystemPromptType,
    options: PromptFormatOptions = {}
  ): PromptMessage[] {
    const {
      maxContextLength = PromptService.DEFAULT_MAX_CONTEXT_LENGTH,
      maxContextMessages = PromptService.DEFAULT_MAX_CONTEXT_MESSAGES,
      contextScoreThreshold = PromptService.DEFAULT_CONTEXT_SCORE_THRESHOLD
    } = options;

    // Фильтруем и сортируем контекст
    const relevantContext = context
      .filter(ctx => ctx.score >= contextScoreThreshold)
      .slice(0, maxContextMessages);

    // Форматируем контекст
    const formattedContext = this.formatContext(relevantContext, maxContextLength);

    // Формируем промпт
    return [
      {
        role: 'system',
        content: SYSTEM_PROMPTS[provider]
      },
      {
        role: 'user',
        content: this.formatUserPrompt(message, formattedContext)
      }
    ];
  }

  /**
   * Форматирование контекста
   */
  private formatContext(
    context: ContextSearchResult[],
    maxLength: number
  ): string {
    let formattedContext = '';
    let currentLength = 0;

    for (const ctx of context) {
      const contextEntry = `\n---\nСообщение (релевантность: ${Math.round(ctx.score * 100)}%):\n${ctx.message.message}`;
      
      // Проверяем, не превысим ли лимит
      if (currentLength + contextEntry.length > maxLength) {
        break;
      }

      formattedContext += contextEntry;
      currentLength += contextEntry.length;
      ctx.usedInPrompt = true;
    }

    return formattedContext;
  }

  /**
   * Форматирование пользоательского промпта с контекстом
   */
  private formatUserPrompt(message: string, context: string): string {
    if (!context) {
      return message;
    }

    return `Контекст для ответа:${context}\n\nВопрос: ${message}`;
  }

  /**
   * Форматирование истории сообщений
   */
  formatMessageHistory(messages: Message[]): PromptMessage[] {
    return messages
      .filter(msg => msg.message || msg.response) // Фильтруем сообщения без текста
      .map(msg => ({
        role: msg.provider === 'system' ? 'system' : 
              msg.provider === 'assistant' ? 'assistant' : 'user',
        content: msg.response || msg.message
      }));
  }
} 