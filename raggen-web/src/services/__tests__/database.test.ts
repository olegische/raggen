import { DatabaseService, CreateMessageParams, CreateContextParams } from '../database';
import { PrismaClient, Message, Context, Chat } from '@prisma/client';

// Мокаем Prisma
jest.mock('@prisma/client', () => {
  const mockPrismaClient = {
    message: {
      create: jest.fn(),
      findUnique: jest.fn(),
      findMany: jest.fn()
    },
    embedding: {
      create: jest.fn(),
      findUnique: jest.fn(),
      findMany: jest.fn()
    },
    context: {
      create: jest.fn(),
      createMany: jest.fn(),
      findMany: jest.fn()
    },
    chat: {
      create: jest.fn(),
      findUnique: jest.fn(),
      findMany: jest.fn()
    },
    $transaction: jest.fn((callback) => callback(mockPrismaClient))
  };

  return {
    PrismaClient: jest.fn(() => mockPrismaClient)
  };
});

describe('DatabaseService', () => {
  let service: DatabaseService;
  let prisma: jest.Mocked<PrismaClient>;

  beforeEach(() => {
    prisma = new PrismaClient() as jest.Mocked<PrismaClient>;
    service = new DatabaseService(prisma);
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Chat operations', () => {
    const mockChat: Chat = {
      id: 'chat-1',
      provider: 'yandex',
      createdAt: new Date(),
      updatedAt: new Date()
    };

    it('should create chat', async () => {
      (prisma.chat.create as jest.Mock).mockResolvedValue(mockChat);

      const result = await service.createChat({ provider: 'yandex' });

      expect(result).toEqual(mockChat);
      expect(prisma.chat.create).toHaveBeenCalledWith({
        data: { provider: 'yandex' }
      });
    });

    it('should get chat by id', async () => {
      (prisma.chat.findUnique as jest.Mock).mockResolvedValue(mockChat);

      const result = await service.getChat('chat-1');

      expect(result).toEqual(mockChat);
      expect(prisma.chat.findUnique).toHaveBeenCalledWith({
        where: { id: 'chat-1' }
      });
    });

    it('should get chats by provider', async () => {
      (prisma.chat.findMany as jest.Mock).mockResolvedValue([mockChat]);

      const result = await service.getChatsByProvider('yandex');

      expect(result).toEqual([mockChat]);
      expect(prisma.chat.findMany).toHaveBeenCalledWith({
        where: { provider: 'yandex' },
        orderBy: { createdAt: 'desc' }
      });
    });
  });

  describe('Message operations', () => {
    const mockMessageParams: CreateMessageParams = {
      chatId: 'chat-1',
      message: 'Test message',
      model: 'test-model',
      provider: 'test-provider',
      temperature: 0.7,
      maxTokens: 1000,
      response: null
    };

    const mockMessage: Message = {
      id: 1,
      ...mockMessageParams,
      timestamp: new Date()
    };

    it('should create message', async () => {
      (prisma.message.create as jest.Mock).mockResolvedValue(mockMessage);

      const result = await service.createMessage(mockMessageParams);

      expect(result).toEqual(mockMessage);
      expect(prisma.message.create).toHaveBeenCalledWith({
        data: mockMessageParams
      });
    });

    it('should get message by id', async () => {
      (prisma.message.findUnique as jest.Mock).mockResolvedValue(mockMessage);

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
      (prisma.message.findMany as jest.Mock).mockResolvedValue([mockMessage]);

      const result = await service.getMessagesByChat('chat-1');

      expect(result).toEqual([mockMessage]);
      expect(prisma.message.findMany).toHaveBeenCalledWith({
        where: { chatId: 'chat-1' },
        include: {
          embedding: true,
          usedContext: true
        },
        orderBy: { timestamp: 'asc' }
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
      (prisma.context.create as jest.Mock).mockResolvedValue(mockContext);

      const result = await service.createContext(mockContextParams);

      expect(result).toEqual(mockContext);
      expect(prisma.context.create).toHaveBeenCalledWith({
        data: mockContextParams
      });
    });

    it('should create many contexts', async () => {
      (prisma.context.createMany as jest.Mock).mockResolvedValue({ count: 1 });

      await service.createManyContexts([mockContextParams]);

      expect(prisma.context.createMany).toHaveBeenCalledWith({
        data: [mockContextParams]
      });
    });

    it('should get context by message id', async () => {
      (prisma.context.findMany as jest.Mock).mockResolvedValue([mockContext]);

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

  describe('Transactions', () => {
    const mockMessageParams: CreateMessageParams = {
      chatId: 'chat-1',
      message: 'Test message',
      model: 'test-model',
      provider: 'test-provider',
      temperature: 0.7,
      maxTokens: 1000,
      response: null
    };

    const mockMessage: Message = {
      id: 1,
      ...mockMessageParams,
      timestamp: new Date()
    };

    it('should create message with embedding in transaction', async () => {
      (prisma.message.create as jest.Mock).mockResolvedValue(mockMessage);
      (prisma.embedding.create as jest.Mock).mockResolvedValue({
        id: 1,
        messageId: 1,
        vector: Buffer.from([]),
        vectorId: 1,
        createdAt: new Date()
      });

      const result = await service.createMessageWithEmbedding(
        mockMessageParams,
        {
          vector: Buffer.from([]),
          vectorId: 1
        }
      );

      expect(result).toEqual(mockMessage);
      expect(prisma.$transaction).toHaveBeenCalled();
    });

    it('should create message with embedding and context in transaction', async () => {
      (prisma.message.create as jest.Mock).mockResolvedValue(mockMessage);
      (prisma.embedding.create as jest.Mock).mockResolvedValue({
        id: 1,
        messageId: 1,
        vector: Buffer.from([]),
        vectorId: 1,
        createdAt: new Date()
      });
      (prisma.context.createMany as jest.Mock).mockResolvedValue({ count: 1 });

      const result = await service.createMessageWithEmbeddingAndContext(
        mockMessageParams,
        {
          vector: Buffer.from([]),
          vectorId: 1
        },
        [{
          sourceId: 2,
          score: 0.9,
          usedInPrompt: true
        }]
      );

      expect(result).toEqual(mockMessage);
      expect(prisma.$transaction).toHaveBeenCalled();
    });
  });
}); 