import { PrismaClient, Message, Embedding, Context } from '@prisma/client';
import { mockDeep, mockReset, DeepMockProxy } from 'jest-mock-extended';
import { DatabaseService, CreateMessageParams, CreateEmbeddingParams, CreateContextParams } from '../database';

// Мокаем PrismaClient
jest.mock('@prisma/client', () => ({
  PrismaClient: jest.fn()
}));

describe('DatabaseService', () => {
  let prisma: DeepMockProxy<PrismaClient>;
  let service: DatabaseService;

  beforeEach(() => {
    prisma = mockDeep<PrismaClient>();
    mockReset(prisma);
    service = new DatabaseService(prisma);
  });

  describe('Message operations', () => {
    const mockMessageParams: CreateMessageParams = {
      chatId: 'test-chat-id',
      message: 'Test message',
      model: 'test-model',
      provider: 'yandex',
      temperature: 0.7,
      maxTokens: 1000
    };

    const mockMessage: Message = {
      id: 1,
      chatId: mockMessageParams.chatId,
      message: mockMessageParams.message,
      response: null,
      model: mockMessageParams.model,
      provider: mockMessageParams.provider,
      temperature: mockMessageParams.temperature,
      maxTokens: mockMessageParams.maxTokens,
      timestamp: new Date()
    };

    it('should create message', async () => {
      prisma.message.create.mockResolvedValue(mockMessage);

      const result = await service.createMessage(mockMessageParams);

      expect(result).toEqual(mockMessage);
      expect(prisma.message.create).toHaveBeenCalledWith({
        data: mockMessageParams
      });
    });

    it('should get message by id with relations', async () => {
      prisma.message.findUnique.mockResolvedValue(mockMessage);

      const result = await service.getMessage(1);

      expect(result).toEqual(mockMessage);
      expect(prisma.message.findUnique).toHaveBeenCalledWith({
        where: { id: 1 },
        include: {
          embedding: true,
          usedContext: true
        }
      });
    });

    it('should get messages by chat id', async () => {
      prisma.message.findMany.mockResolvedValue([mockMessage]);

      const result = await service.getMessagesByChat('test-chat-id');

      expect(result).toEqual([mockMessage]);
      expect(prisma.message.findMany).toHaveBeenCalledWith({
        where: { chatId: 'test-chat-id' },
        include: {
          embedding: true,
          usedContext: true
        },
        orderBy: { timestamp: 'asc' }
      });
    });
  });

  describe('Embedding operations', () => {
    const mockEmbeddingParams: CreateEmbeddingParams = {
      messageId: 1,
      vector: Buffer.from('test'),
      vectorId: 100
    };

    const mockEmbedding: Embedding = {
      id: 1,
      messageId: mockEmbeddingParams.messageId,
      vector: mockEmbeddingParams.vector,
      vectorId: mockEmbeddingParams.vectorId,
      createdAt: new Date()
    };

    it('should create embedding', async () => {
      prisma.embedding.create.mockResolvedValue(mockEmbedding);

      const result = await service.createEmbedding(mockEmbeddingParams);

      expect(result).toEqual(mockEmbedding);
      expect(prisma.embedding.create).toHaveBeenCalledWith({
        data: mockEmbeddingParams
      });
    });

    it('should get embedding by message id', async () => {
      prisma.embedding.findUnique.mockResolvedValue(mockEmbedding);

      const result = await service.getEmbeddingByMessageId(1);

      expect(result).toEqual(mockEmbedding);
      expect(prisma.embedding.findUnique).toHaveBeenCalledWith({
        where: { messageId: 1 }
      });
    });

    it('should get embeddings by vector ids', async () => {
      prisma.embedding.findMany.mockResolvedValue([mockEmbedding]);

      const result = await service.getEmbeddingsByVectorIds([100]);

      expect(result).toEqual([mockEmbedding]);
      expect(prisma.embedding.findMany).toHaveBeenCalledWith({
        where: {
          vectorId: {
            in: [100]
          }
        },
        include: {
          message: true
        }
      });
    });
  });

  describe('Context operations', () => {
    const mockContextParams: CreateContextParams = {
      messageId: 1,
      sourceId: 2,
      score: 0.9,
      usedInPrompt: true
    };

    const mockContext: Context = {
      id: 1,
      messageId: mockContextParams.messageId,
      sourceId: mockContextParams.sourceId,
      score: mockContextParams.score,
      usedInPrompt: mockContextParams.usedInPrompt,
      createdAt: new Date()
    };

    it('should create context', async () => {
      prisma.context.create.mockResolvedValue(mockContext);

      const result = await service.createContext(mockContextParams);

      expect(result).toEqual(mockContext);
      expect(prisma.context.create).toHaveBeenCalledWith({
        data: mockContextParams
      });
    });

    it('should create many contexts', async () => {
      prisma.context.createMany.mockResolvedValue({ count: 1 });

      await service.createManyContexts([mockContextParams]);

      expect(prisma.context.createMany).toHaveBeenCalledWith({
        data: [mockContextParams]
      });
    });

    it('should get context by message id', async () => {
      prisma.context.findMany.mockResolvedValue([mockContext]);

      const result = await service.getContextByMessageId(1);

      expect(result).toEqual([mockContext]);
      expect(prisma.context.findMany).toHaveBeenCalledWith({
        where: { messageId: 1 },
        include: {
          message: true
        },
        orderBy: { score: 'desc' }
      });
    });
  });

  describe('Transaction operations', () => {
    const mockMessageParams: CreateMessageParams = {
      chatId: 'test-chat-id',
      message: 'Test message',
      model: 'test-model',
      provider: 'yandex',
      temperature: 0.7,
      maxTokens: 1000
    };

    const mockMessage: Message = {
      id: 1,
      chatId: mockMessageParams.chatId,
      message: mockMessageParams.message,
      response: null,
      model: mockMessageParams.model,
      provider: mockMessageParams.provider,
      temperature: mockMessageParams.temperature,
      maxTokens: mockMessageParams.maxTokens,
      timestamp: new Date()
    };

    const mockEmbeddingParams: Omit<CreateEmbeddingParams, 'messageId'> = {
      vector: Buffer.from('test'),
      vectorId: 100
    };

    const mockContextParams: Omit<CreateContextParams, 'messageId'> = {
      sourceId: 2,
      score: 0.9,
      usedInPrompt: true
    };

    it('should create message with embedding in transaction', async () => {
      const mockTransaction = mockDeep<PrismaClient>();
      prisma.$transaction.mockImplementation(async (callback: any) => {
        if (typeof callback === 'function') {
          return callback(mockTransaction);
        }
        return mockTransaction;
      });
      mockTransaction.message.create.mockResolvedValue(mockMessage);

      const result = await service.createMessageWithEmbedding(
        mockMessageParams,
        mockEmbeddingParams
      );

      expect(result).toEqual(mockMessage);
      expect(mockTransaction.message.create).toHaveBeenCalled();
      expect(mockTransaction.embedding.create).toHaveBeenCalled();
    });

    it('should create message with embedding and context in transaction', async () => {
      const mockTransaction = mockDeep<PrismaClient>();
      prisma.$transaction.mockImplementation(async (callback: any) => {
        if (typeof callback === 'function') {
          return callback(mockTransaction);
        }
        return mockTransaction;
      });
      mockTransaction.message.create.mockResolvedValue(mockMessage);

      const result = await service.createMessageWithEmbeddingAndContext(
        mockMessageParams,
        mockEmbeddingParams,
        [mockContextParams]
      );

      expect(result).toEqual(mockMessage);
      expect(mockTransaction.message.create).toHaveBeenCalled();
      expect(mockTransaction.embedding.create).toHaveBeenCalled();
      expect(mockTransaction.context.createMany).toHaveBeenCalled();
    });
  });
}); 