import { ContextService, ContextSearchResult } from '../context.service';
import { DatabaseService } from '../database';
import { EmbedApiClient } from '../embed-api';
import { Message } from '@prisma/client';

// Мокаем зависимости
jest.mock('../database');
jest.mock('../embed-api');

describe('ContextService', () => {
  let contextService: ContextService;
  let database: jest.Mocked<DatabaseService>;
  let embedApi: jest.Mocked<EmbedApiClient>;

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

  const mockEmbedding = {
    id: 1,
    messageId: 1,
    vector: Buffer.from(new Float32Array(10).buffer),
    vectorId: 1,
    createdAt: new Date()
  };

  beforeEach(() => {
    database = {
      getEmbeddingsByVectorIds: jest.fn(),
      createManyContexts: jest.fn(),
      getMessagesByIds: jest.fn()
    } as unknown as jest.Mocked<DatabaseService>;

    embedApi = {
      searchSimilar: jest.fn(),
      embedText: jest.fn()
    } as unknown as jest.Mocked<EmbedApiClient>;

    contextService = new ContextService(database, embedApi);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('searchContext', () => {
    const text = 'test query';
    const mockSearchResponse = {
      query: text,
      results: [
        { vector_id: 1, score: 0.9, text: 'Context message 1' },
        { vector_id: 2, score: 0.8, text: 'Context message 2' }
      ]
    };

    beforeEach(() => {
      embedApi.searchSimilar.mockResolvedValue(mockSearchResponse);
      database.getEmbeddingsByVectorIds.mockResolvedValue([
        { ...mockEmbedding, messageId: mockMessage.id }
      ]);
      database.getMessagesByIds.mockResolvedValue([mockMessage]);
    });

    it('should search for context using embeddings API', async () => {
      await contextService.searchContext(text);

      expect(embedApi.searchSimilar).toHaveBeenCalledWith(text, 10);
    });

    it('should filter results by minimum score', async () => {
      const results = await contextService.searchContext('test', { minScore: 0.85 });
      expect(results.length).toBe(1);
      expect(results[0].score).toBe(0.9);
    });

    it('should exclude specified message IDs', async () => {
      await contextService.searchContext('test', { excludeMessageIds: [1] });
      expect(database.getEmbeddingsByVectorIds).toHaveBeenCalledWith([2]);
    });

    it('should limit number of results', async () => {
      await contextService.searchContext('test', { maxResults: 1 });
      expect(database.getEmbeddingsByVectorIds).toHaveBeenCalledWith([1]);
    });

    it('should return cached results if available', async () => {
      const text = 'test query';
      
      // Первый апрос - без кэша
      await contextService.searchContext(text);
      expect(embedApi.searchSimilar).toHaveBeenCalledTimes(1);

      // Второй запрос - должен использовать кэш
      await contextService.searchContext(text);
      expect(embedApi.searchSimilar).toHaveBeenCalledTimes(1);
    });
  });

  describe('saveUsedContext', () => {
    const mockContexts: ContextSearchResult[] = [
      {
        message: mockMessage,
        score: 0.9,
        usedInPrompt: true
      }
    ];

    it('should save context to database', async () => {
      await contextService.saveUsedContext(1, mockContexts);

      expect(database.createManyContexts).toHaveBeenCalledWith([
        {
          messageId: 1,
          sourceId: mockMessage.id,
          score: 0.9,
          usedInPrompt: true
        }
      ]);
    });
  });

  describe('prioritizeResults', () => {
    const text = 'test query';
    const message1 = { ...mockMessage, id: 1, score: 0.9 };
    const message2 = { ...mockMessage, id: 2, score: 0.7 };

    beforeEach(() => {
      embedApi.searchSimilar.mockResolvedValue({
        query: text,
        results: [
          { vector_id: 1, score: 0.9, text: 'Context message 1' },
          { vector_id: 2, score: 0.8, text: 'Context message 2' }
        ]
      });
    });

    it('should prioritize by score when difference is significant', async () => {
      database.getEmbeddingsByVectorIds.mockResolvedValue([
        { ...mockEmbedding, vectorId: 1, messageId: message1.id },
        { ...mockEmbedding, vectorId: 2, messageId: message2.id }
      ]);
      database.getMessagesByIds.mockResolvedValue([message1, message2]);

      const results = await contextService.searchContext(text);

      expect(results[0].message.id).toBe(message1.id);
      expect(results[1].message.id).toBe(message2.id);
    });

    it('should prioritize by timestamp when scores are close', async () => {
      const olderMessage = { ...mockMessage, id: 1, timestamp: new Date('2023-01-01') };
      const newerMessage = { ...mockMessage, id: 2, timestamp: new Date('2023-01-02') };

      database.getEmbeddingsByVectorIds.mockResolvedValue([
        { ...mockEmbedding, vectorId: 1, messageId: olderMessage.id },
        { ...mockEmbedding, vectorId: 2, messageId: newerMessage.id }
      ]);
      database.getMessagesByIds.mockResolvedValue([olderMessage, newerMessage]);

      const results = await contextService.searchContext(text);

      expect(results[0].message.id).toBe(newerMessage.id);
      expect(results[1].message.id).toBe(olderMessage.id);
    });
  });
}); 