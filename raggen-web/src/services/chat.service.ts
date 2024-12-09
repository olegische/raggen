import { ProviderFactory, ProviderType } from '@/providers/factory';
import { GenerationOptions } from '@/providers/base.provider';
import { GENERATION_CONFIG } from '@/config/generation';
import { DatabaseService } from './database';
import { ContextService } from './context.service';
import { PromptService } from './prompt.service';
import { EmbedApiClient } from './embed-api';
import { SystemPromptType } from '@/config/prompts';

export interface SendMessageOptions extends GenerationOptions {
  maxContextMessages?: number;
  contextScoreThreshold?: number;
}

export class ChatService {
  private provider;
  private database: DatabaseService;
  private contextService: ContextService;
  private promptService: PromptService;

  constructor(
    providerType: ProviderType,
    database?: DatabaseService,
    embedApi?: EmbedApiClient
  ) {
    this.provider = ProviderFactory.createProvider(providerType);
    this.database = database || new DatabaseService();
    this.contextService = new ContextService(
      this.database,
      embedApi || new EmbedApiClient()
    );
    this.promptService = new PromptService();
  }

  async sendMessage(
    message: string,
    chatId?: string,
    options?: SendMessageOptions
  ) {
    try {
      // Получаем или создаем чат
      const chat = chatId 
        ? await this.database.getChat(chatId)
        : await this.database.createChat({
            provider: this.provider.constructor.name.replace('Provider', '').toLowerCase()
          });

      if (!chat) {
        throw new Error('Chat not found');
      }

      // Получаем предыдущие сообщения для контекста
      const previousMessages = await this.database.getMessagesByChat(chat.id);

      // Ищем релевантный контекст
      const context = await this.contextService.searchContext(message, {
        maxResults: options?.maxContextMessages,
        minScore: options?.contextScoreThreshold,
        excludeMessageIds: previousMessages.map(msg => msg.id)
      });

      // Форматируем промпт с контекстом
      const promptMessages = this.promptService.formatPromptWithContext(
        message,
        context,
        this.provider.constructor.name.replace('Provider', '').toLowerCase() as SystemPromptType,
        {
          maxContextMessages: options?.maxContextMessages,
          contextScoreThreshold: options?.contextScoreThreshold
        }
      );

      // Добавляем историю сообщений
      const historyMessages = this.promptService.formatMessageHistory(previousMessages);
      promptMessages.splice(1, 0, ...historyMessages);

      // Получаем ответ от провайдера
      const response = await this.provider.generateResponse(
        message,
        options,
        promptMessages
      );

      // Получаем эмбеддинг
      const embedResponse = await this.contextService.embedApi.embedText(message);

      // Создаем сообщение и эмбеддинг в одной транзакции
      const savedMessage = await this.database.createMessageWithEmbedding(
        {
          chatId: chat.id,
          message: message,
          response: response.text,
          model: options?.model || 'default',
          provider: this.provider.constructor.name.replace('Provider', '').toLowerCase(),
          temperature: options?.temperature || GENERATION_CONFIG.temperature.default,
          maxTokens: options?.maxTokens || GENERATION_CONFIG.maxTokens.default
        },
        {
          vector: Buffer.from(new Float32Array(embedResponse.embedding).buffer),
          vectorId: embedResponse.vector_id
        }
      );

      // Сохраняем использованный контекст через context service
      if (context.length > 0) {
        await this.contextService.saveUsedContext(savedMessage.id, context);
      }

      return {
        message: savedMessage,
        context,
        chatId: chat.id
      };

    } catch (error) {
      console.error('Error in ChatService:', error);
      throw error;
    }
  }

  async getHistory(chatId: string) {
    return this.database.getMessagesByChat(chatId);
  }
} 