import { GenerationOptions } from '../providers/base.provider';
import { GENERATION_CONFIG } from '../config/generation';
import { ContextService } from './context.service';
import { PromptService } from './prompt.service';
import { EmbedApiClient } from './embed-api';
import { ProviderFactory, ProviderType } from '../providers/factory';
import { ChatRepository } from './repositories/chat.repository';
import { EmbeddingRepository } from './repositories/embedding.repository';
import { BaseRepository } from './repositories/base.repository';

export interface SendMessageOptions extends GenerationOptions {
  maxContextMessages?: number;
  contextScoreThreshold?: number;
}

export class ChatService {
  private provider;
  private chatRepository: ChatRepository;
  private embeddingRepository: EmbeddingRepository;
  private contextService: ContextService;
  private promptService: PromptService;
  private providerType: ProviderType;

  constructor(
    providerType: ProviderType,
    baseRepository?: BaseRepository,
    embedApi?: EmbedApiClient
  ) {
    const repo = baseRepository || new BaseRepository();
    const api = embedApi || new EmbedApiClient();

    this.providerType = providerType;
    this.provider = ProviderFactory.getProvider(providerType);
    this.chatRepository = repo.createRepository(ChatRepository);
    this.embeddingRepository = repo.createRepository(EmbeddingRepository);
    this.contextService = new ContextService(repo, api);
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
        ? await this.chatRepository.getChat(chatId)
        : await this.chatRepository.createChat({
            provider: this.providerType
          });

      if (!chat) {
        throw new Error('Chat not found');
      }

      // Получаем предыдущие сообщения для контекста
      const previousMessages = await this.chatRepository.getMessagesByChat(chat.id);

      // Ищем релевантный контекст
      const context = await this.contextService.searchContext(message, {
        maxResults: options?.maxContextMessages,
        minScore: options?.contextScoreThreshold,
        excludeMessageIds: previousMessages.map(msg => msg.id)
      });

      // Получаем информацию о провайдере из фабрики
      const providerInfo = ProviderFactory.getSupportedProviders()
        .find(p => p.id === this.providerType);

      if (!providerInfo) {
        throw new Error('Provider not found');
      }

      // Форматируем промпт с контекстом
      const promptMessages = this.promptService.formatPromptWithContext(
        message,
        context,
        providerInfo.systemPrompt,
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

      // Создаем сообщение с эмбеддингом
      const savedMessage = await this.chatRepository.createMessageWithEmbedding(
        {
          chatId: chat.id,
          message: message,
          response: response.text,
          model: options?.model || 'default',
          provider: this.providerType,
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
    return this.chatRepository.getMessagesByChat(chatId);
  }
}
