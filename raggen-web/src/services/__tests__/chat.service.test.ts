import { ChatService } from '../chat.service';
import { DatabaseService } from '../database';
import { EmbedApiClient } from '../embed-api';
import { Message, Chat } from '@prisma/client';
import { ProviderFactory } from '@/providers/factory';
import { BaseProvider } from '@/providers/base.provider';

// Мокаем зависимости
jest.mock('../database');
jest.mock('../embed-api');
jest.mock('@/providers/factory', () => ({
  ProviderFactory: {
    createProvider: jest.fn(() => ({
      generateResponse: jest.fn().mockResolvedValue({
        text: 'Mock response',
        usage: { prompt_tokens: 10, completion_tokens: 20 }
      })
    }))
  }
}));

describe('ChatService', () => {
  let chatService: ChatService;
  let database: jest.Mocked<DatabaseService>;
  let embedApi: jest.Mocked<EmbedApiClient>;
  let provider: jest.Mocked<BaseProvider>;

  const mockChat: Chat = {
    id: 'chat-1',
    provider: 'yandex',
    createdAt: new Date(),
    updatedAt: new Date()
  };

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
    database = {
      getChat: jest.fn(),
      createChat: jest.fn(),
      getMessagesByChat: jest.fn(),
      createMessageWithEmbedding: jest.fn(),
      getEmbeddingsByVectorIds: jest.fn(),
      getMessagesByIds: jest.fn(),
      createManyContexts: jest.fn()
    } as unknown as jest.Mocked<DatabaseService>;

    embedApi = {
      searchSimilar: jest.fn(),
      embedText: jest.fn()
    } as unknown as jest.Mocked<EmbedApiClient>;

    provider = {
      generateResponse: jest.fn()
    } as unknown as jest.Mocked<BaseProvider>;

    ProviderFactory.createProvider = jest.fn().mockReturnValue(provider);

    chatService = new ChatService('yandex', database, embedApi);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('sendMessage', () => {
    const message = 'Test message';
    const chatId = 'chat-1';
    const mockEmbedding = {
      id: 1,
      messageId: 1,
      vector: Buffer.from(new Float32Array(10).buffer),
      vectorId: 1,
      createdAt: new Date()
    };

    beforeEach(() => {
      database.getChat.mockResolvedValue(mockChat);
      database.createChat.mockResolvedValue(mockChat);
      database.getMessagesByChat.mockResolvedValue([mockMessage]);
      database.createMessageWithEmbedding.mockResolvedValue(mockMessage);
      database.getEmbeddingsByVectorIds.mockResolvedValue([mockEmbedding]);
      database.getMessagesByIds.mockResolvedValue([mockMessage]);
      database.createManyContexts.mockResolvedValue();

      embedApi.searchSimilar.mockResolvedValue({
        query: message,
        results: [
          { vector_id: 1, score: 0.9, text: 'Context message 1' }
        ]
      });
      embedApi.embedText.mockResolvedValue({
        embedding: Array.from(new Float32Array(10)),
        vector_id: 1,
        text: message
      });

      provider.generateResponse.mockResolvedValue({
        text: 'Response text',
        usage: {
          promptTokens: 50,
          completionTokens: 50,
          totalTokens: 100
        }
      });
    });

    it('should send message and save context', async () => {
      const result = await chatService.sendMessage(message, chatId);

      expect(result.message).toEqual(mockMessage);
      expect(result.chatId).toBe(chatId);
      expect(database.createMessageWithEmbedding).toHaveBeenCalled();
      expect(database.createManyContexts).toHaveBeenCalled();
    });

    it('should create new chat if chatId not provided', async () => {
      database.getChat.mockResolvedValue(null);
      const result = await chatService.sendMessage(message);

      expect(database.createChat).toHaveBeenCalled();
      expect(result.chatId).toBe(mockChat.id);
    });

    it('should use context options if provided', async () => {
      const options = {
        maxContextMessages: 2,
        contextScoreThreshold: 0.8
      };

      await chatService.sendMessage(message, chatId, options);

      expect(embedApi.searchSimilar).toHaveBeenCalled();
    });

    it('should handle database errors', async () => {
      database.createMessageWithEmbedding.mockRejectedValue(new Error('Database error'));

      await expect(chatService.sendMessage(message, chatId))
        .rejects.toThrow('Database error');
    });

    it('should handle missing chat', async () => {
      database.getChat.mockResolvedValue(null);

      await expect(chatService.sendMessage(message, chatId))
        .rejects.toThrow('Chat not found');
    });
  });

  describe('getHistory', () => {
    it('should return chat history', async () => {
      database.getMessagesByChat.mockResolvedValue([mockMessage]);

      const history = await chatService.getHistory('chat-1');

      expect(history).toEqual([mockMessage]);
      expect(database.getMessagesByChat).toHaveBeenCalledWith('chat-1');
    });
  });
}); 