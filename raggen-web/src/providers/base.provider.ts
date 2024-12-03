import { z } from 'zod';
import { SYSTEM_PROMPTS, SystemPromptType } from '@/config/prompts';

// Общие типы для всех провайдеров
export interface ProviderConfig {
  apiUrl: string;
  credentials: string;
  systemPrompt?: SystemPromptType;
}

export interface ProviderMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface GenerationOptions {
  temperature?: number;
  maxTokens?: number;
  model?: string;
}

export interface GenerationResult {
  text: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

// Базовый класс для всех провайдеров
export abstract class BaseProvider {
  protected config: ProviderConfig;
  protected responseSchema: z.ZodType;
  protected systemPrompt: string;

  constructor(config: ProviderConfig, responseSchema: z.ZodType) {
    this.config = config;
    this.responseSchema = responseSchema;
    this.systemPrompt = SYSTEM_PROMPTS[config.systemPrompt || 'default'];
  }

  // Абстрактные методы, которые должны быть реализованы в каждом провайдере
  abstract generateResponse(
    message: string,
    options?: GenerationOptions,
    previousMessages?: ProviderMessage[]
  ): Promise<GenerationResult>;

  abstract listModels(): Promise<string[]>;

  // Общие методы для всех провайдеров
  protected validateResponse(response: unknown): unknown {
    return this.responseSchema.parse(response);
  }

  protected formatError(error: unknown): Error {
    if (error instanceof Error) {
      return error;
    }
    return new Error('Unknown error occurred');
  }

  protected validateOptions(options?: GenerationOptions): GenerationOptions {
    return {
      temperature: options?.temperature ?? 0.7,
      maxTokens: options?.maxTokens ?? 1000
    };
  }

  protected formatMessages(message: string, previousMessages?: ProviderMessage[]): ProviderMessage[] {
    const messages: ProviderMessage[] = [
      { role: 'system', content: this.systemPrompt }
    ];

    if (previousMessages) {
      messages.push(...previousMessages);
    }

    messages.push({ role: 'user', content: message });
    return messages;
  }
} 