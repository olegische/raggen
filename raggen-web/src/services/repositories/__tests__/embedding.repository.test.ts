import { PrismaClient } from '@prisma/client';
import { EmbeddingRepository } from '../embedding.repository';
import { BaseRepository } from '../base.repository';
import '@testing-library/jest-dom';

describe('EmbeddingRepository', () => {
  let repository: EmbeddingRepository;
  let baseRepository: BaseRepository;
  let prisma: PrismaClient;

  beforeAll(async () => {
    prisma = new PrismaClient();
    await prisma.$connect();
  });

  afterAll(async () => {
    await prisma.$disconnect();
  });

  beforeEach(() => {
    baseRepository = new BaseRepository(prisma);
    repository = baseRepository.createRepository(EmbeddingRepository);
  });

  afterEach(async () => {
    // Очищаем тестовые данные
    await prisma.context.deleteMany();
    await prisma.embedding.deleteMany();
    await prisma.message.deleteMany();
    await prisma.chat.deleteMany();
  });

  describe('Embedding operations', () => {
    const mockMessage = {
      chatId: 'test-chat-id',
      message: 'Test message',
      model: 'test-model',
      provider: 'test-provider',
      temperature: 0.7,
      maxTokens: 1000,
      response: null
    };

    const mockEmbedding = {
      messageId: 1,
      vector: Buffer.from([1, 2, 3, 4]),
      vectorId: 1
    };

    it('should create embedding with valid params', async () => {
      // Создаем чат и сообщение для теста
      const chat = await prisma.chat.create({
        data: {
          id: 'test-chat-id',
          provider: 'test'
        }
      });

      const message = await prisma.message.create({
        data: mockMessage
      });

      const embedding = await repository.createEmbedding({
        ...mockEmbedding,
        messageId: message.id
      });

      expect(embedding).toBeDefined();
      expect(embedding.messageId).toBe(message.id);
      expect(embedding.vector).toEqual(mockEmbedding.vector);
      expect(embedding.vectorId).toBe(mockEmbedding.vectorId);
      expect(embedding.id).toBeDefined();
      expect(embedding.createdAt).toBeDefined();
    });

    it('should throw error for invalid message ID', async () => {
      await expect(repository.createEmbedding({
        ...mockEmbedding,
        messageId: 0
      })).rejects.toThrow('Invalid message ID');
    });

    it('should throw error for empty vector', async () => {
      await expect(repository.createEmbedding({
        ...mockEmbedding,
        vector: Buffer.from([])
      })).rejects.toThrow('Vector is required');
    });

    it('should throw error for invalid vector ID', async () => {
      await expect(repository.createEmbedding({
        ...mockEmbedding,
        vectorId: -1
      })).rejects.toThrow('Invalid vector ID');
    });

    it('should get embedding by valid message id', async () => {
      // Создаем чат и сообщение для теста
      const chat = await prisma.chat.create({
        data: {
          id: 'test-chat-id',
          provider: 'test'
        }
      });

      const message = await prisma.message.create({
        data: mockMessage
      });

      const created = await prisma.embedding.create({
        data: {
          ...mockEmbedding,
          messageId: message.id
        }
      });

      const found = await repository.getEmbeddingByMessageId(message.id);

      expect(found).toBeDefined();
      expect(found?.id).toBe(created.id);
      expect(found?.messageId).toBe(message.id);
      expect(found?.vector).toEqual(mockEmbedding.vector);
    });

    it('should throw error for invalid message ID', async () => {
      await expect(repository.getEmbeddingByMessageId(0))
        .rejects.toThrow('Invalid message ID');
    });

    it('should return null when embedding not found', async () => {
      const found = await repository.getEmbeddingByMessageId(999);
      expect(found).toBeNull();
    });

    it('should get embeddings by valid vector ids', async () => {
      // Создаем чат и сообщения для теста
      const chat = await prisma.chat.create({
        data: {
          id: 'test-chat-id',
          provider: 'test'
        }
      });

      const message1 = await prisma.message.create({
        data: mockMessage
      });

      const message2 = await prisma.message.create({
        data: mockMessage
      });

      const embedding1 = await prisma.embedding.create({
        data: {
          ...mockEmbedding,
          messageId: message1.id,
          vectorId: 1
        }
      });

      const embedding2 = await prisma.embedding.create({
        data: {
          ...mockEmbedding,
          messageId: message2.id,
          vectorId: 2
        }
      });

      const embeddings = await repository.getEmbeddingsByVectorIds([1, 2]);

      expect(embeddings).toHaveLength(2);
      expect(embeddings.map(e => e.id)).toContain(embedding1.id);
      expect(embeddings.map(e => e.id)).toContain(embedding2.id);
      // Проверяем, что включены связи с сообщениями
      expect(embeddings[0].message).toBeDefined();
      expect(embeddings[1].message).toBeDefined();
    });

    it('should throw error for empty vector IDs array', async () => {
      await expect(repository.getEmbeddingsByVectorIds([]))
        .rejects.toThrow('Vector IDs array is empty');
    });

    it('should throw error for invalid vector IDs', async () => {
      await expect(repository.getEmbeddingsByVectorIds([1, -1]))
        .rejects.toThrow('Invalid vector ID in array');
    });
  });

  describe('Context operations', () => {
    const mockMessage = {
      chatId: 'test-chat-id',
      message: 'Test message',
      model: 'test-model',
      provider: 'test-provider',
      temperature: 0.7,
      maxTokens: 1000,
      response: null
    };

    const mockContext = {
      messageId: 1,
      sourceId: 2,
      score: 0.9,
      usedInPrompt: true
    };

    it('should create context with valid params', async () => {
      // Создаем чат и сообщения для теста
      const chat = await prisma.chat.create({
        data: {
          id: 'test-chat-id',
          provider: 'test'
        }
      });

      const message1 = await prisma.message.create({
        data: mockMessage
      });

      const message2 = await prisma.message.create({
        data: mockMessage
      });

      const context = await repository.createContext({
        ...mockContext,
        messageId: message1.id,
        sourceId: message2.id
      });

      expect(context).toBeDefined();
      expect(context.messageId).toBe(message1.id);
      expect(context.sourceId).toBe(message2.id);
      expect(context.score).toBe(mockContext.score);
      expect(context.usedInPrompt).toBe(mockContext.usedInPrompt);
      expect(context.id).toBeDefined();
      expect(context.createdAt).toBeDefined();
    });

    it('should throw error for invalid message ID', async () => {
      await expect(repository.createContext({
        ...mockContext,
        messageId: 0
      })).rejects.toThrow('Invalid message ID');
    });

    it('should throw error for invalid source ID', async () => {
      await expect(repository.createContext({
        ...mockContext,
        sourceId: -1
      })).rejects.toThrow('Invalid source ID');
    });

    it('should throw error for invalid score', async () => {
      await expect(repository.createContext({
        ...mockContext,
        score: 1.5
      })).rejects.toThrow('Score must be between 0 and 1');
    });

    it('should throw error for invalid usedInPrompt', async () => {
      await expect(repository.createContext({
        ...mockContext,
        usedInPrompt: 'true' as any
      })).rejects.toThrow('usedInPrompt must be a boolean');
    });

    it('should create multiple contexts with valid params', async () => {
      // Создаем чат и сообщения для теста
      const chat = await prisma.chat.create({
        data: {
          id: 'test-chat-id',
          provider: 'test'
        }
      });

      const message1 = await prisma.message.create({
        data: mockMessage
      });

      const message2 = await prisma.message.create({
        data: mockMessage
      });

      const message3 = await prisma.message.create({
        data: mockMessage
      });

      const contexts = [
        {
          ...mockContext,
          messageId: message1.id,
          sourceId: message2.id
        },
        {
          ...mockContext,
          messageId: message1.id,
          sourceId: message3.id
        }
      ];

      await repository.createManyContexts(contexts);

      const found = await repository.getContextByMessageId(message1.id);
      expect(found).toHaveLength(2);
      expect(found.map(c => c.sourceId)).toContain(message2.id);
      expect(found.map(c => c.sourceId)).toContain(message3.id);
    });

    it('should throw error for empty contexts array', async () => {
      await expect(repository.createManyContexts([]))
        .rejects.toThrow('Contexts array is empty');
    });

    it('should throw error if any context is invalid', async () => {
      await expect(repository.createManyContexts([
        mockContext,
        { ...mockContext, score: 1.5 }
      ])).rejects.toThrow('Score must be between 0 and 1');
    });

    it('should get contexts by valid message id', async () => {
      // Создаем чат и сообщения для теста
      const chat = await prisma.chat.create({
        data: {
          id: 'test-chat-id',
          provider: 'test'
        }
      });

      const message1 = await prisma.message.create({
        data: mockMessage
      });

      const message2 = await prisma.message.create({
        data: mockMessage
      });

      const created = await prisma.context.create({
        data: {
          ...mockContext,
          messageId: message1.id,
          sourceId: message2.id
        }
      });

      const found = await repository.getContextByMessageId(message1.id);

      expect(found).toHaveLength(1);
      expect(found[0].id).toBe(created.id);
      expect(found[0].messageId).toBe(message1.id);
      expect(found[0].sourceId).toBe(message2.id);
      expect(found[0].score).toBe(mockContext.score);
      // Проверяем, что включены связи с сообщениями
      expect(found[0].message).toBeDefined();
    });

    it('should throw error for invalid message ID', async () => {
      await expect(repository.getContextByMessageId(0))
        .rejects.toThrow('Invalid message ID');
    });

    it('should return empty array when no contexts found', async () => {
      const found = await repository.getContextByMessageId(999);
      expect(found).toHaveLength(0);
    });

    it('should sort contexts by score in descending order', async () => {
      // Создаем чат и сообщения для теста
      const chat = await prisma.chat.create({
        data: {
          id: 'test-chat-id',
          provider: 'test'
        }
      });

      const message1 = await prisma.message.create({
        data: mockMessage
      });

      const message2 = await prisma.message.create({
        data: mockMessage
      });

      await prisma.context.createMany({
        data: [
          {
            messageId: message1.id,
            sourceId: message2.id,
            score: 0.8,
            usedInPrompt: true
          },
          {
            messageId: message1.id,
            sourceId: message2.id,
            score: 0.9,
            usedInPrompt: true
          },
          {
            messageId: message1.id,
            sourceId: message2.id,
            score: 0.7,
            usedInPrompt: true
          }
        ]
      });

      const contexts = await repository.getContextByMessageId(message1.id);

      expect(contexts).toHaveLength(3);
      expect(contexts[0].score).toBe(0.9);
      expect(contexts[1].score).toBe(0.8);
      expect(contexts[2].score).toBe(0.7);
    });
  });
});
