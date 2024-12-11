import { PrismaClient, Chat, Message, Embedding, Context } from '@prisma/client';
import { ChatRepository } from '../chat.repository';
import { BaseRepository } from '../base.repository';
import '@testing-library/jest-dom';

type MessageWithRelations = Message & {
  embedding?: Embedding | null;
  usedContext?: Context[];
};

// Увеличиваем тайм-аут для тестов с транзакциями
jest.setTimeout(10000);

describe('ChatRepository', () => {
  let repository: ChatRepository;
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
    repository = baseRepository.createRepository(ChatRepository);
  });

  afterEach(async () => {
    // Очищаем тестовые данные
    await prisma.context.deleteMany();
    await prisma.embedding.deleteMany();
    await prisma.message.deleteMany();
    await prisma.chat.deleteMany();
  });

  describe('Chat operations', () => {
    const mockChat = {
      provider: 'test-provider'
    };

    describe('createChat', () => {
      it('should create chat with valid params', async () => {
        const chat = await repository.createChat(mockChat);

        expect(chat).toBeDefined();
        expect(chat.provider).toBe(mockChat.provider);
        expect(chat.id).toBeDefined();
        expect(chat.createdAt).toBeDefined();
        expect(chat.updatedAt).toBeDefined();
      });

      it('should throw error for empty provider', async () => {
        await expect(repository.createChat({
          provider: ''
        })).rejects.toThrow('Provider is required');
      });
    });

    describe('getChat', () => {
      it('should get chat by valid id', async () => {
        const created = await prisma.chat.create({
          data: mockChat
        });

        const chat = await repository.getChat(created.id);

        expect(chat).toBeDefined();
        expect(chat?.id).toBe(created.id);
        expect(chat?.provider).toBe(mockChat.provider);
      });

      it('should throw error for invalid chat ID', async () => {
        await expect(repository.getChat(''))
          .rejects.toThrow('Chat ID is required');
      });

      it('should return null when chat not found', async () => {
        const chat = await repository.getChat('non-existent-id');
        expect(chat).toBeNull();
      });
    });

    describe('getChatsByProvider', () => {
      it('should get chats by valid provider', async () => {
        const chat1 = await prisma.chat.create({
          data: mockChat
        });

        const chat2 = await prisma.chat.create({
          data: mockChat
        });

        const chats = await repository.getChatsByProvider(mockChat.provider);

        expect(chats).toHaveLength(2);
        expect(chats.map(c => c.id)).toContain(chat1.id);
        expect(chats.map(c => c.id)).toContain(chat2.id);
        // Проверяем сортировку по createdAt desc
        expect(chats[0].createdAt.getTime()).toBeGreaterThanOrEqual(chats[1].createdAt.getTime());
      });

      it('should throw error for empty provider', async () => {
        await expect(repository.getChatsByProvider(''))
          .rejects.toThrow('Provider is required');
      });

      it('should return empty array when no chats found', async () => {
        const chats = await repository.getChatsByProvider('non-existent-provider');
        expect(chats).toHaveLength(0);
      });
    });
  });

  describe('Transaction operations', () => {
    const mockChat = {
      provider: 'test-provider'
    };

    const mockMessage = {
      message: 'Test message',
      model: 'test-model',
      provider: 'test-provider',
      temperature: 0.7,
      maxTokens: 1000,
      response: null
    };

    const mockEmbedding = {
      vector: Buffer.from([1, 2, 3, 4]),
      vectorId: 1
    };

    const mockContext = {
      score: 0.9,
      usedInPrompt: true
    };

    describe('createMessageWithEmbedding', () => {
      it('should create message with embedding in transaction', async () => {
        // Создаем чат для теста
        const chat = await prisma.chat.create({
          data: mockChat
        });

        const message = await repository.createMessageWithEmbedding(
          {
            ...mockMessage,
            chatId: chat.id
          },
          mockEmbedding
        ) as MessageWithRelations;

        expect(message).toBeDefined();
        expect(message.chatId).toBe(chat.id);
        expect(message.message).toBe(mockMessage.message);
        expect(message.embedding).toBeDefined();
        expect(message.embedding?.vector).toEqual(mockEmbedding.vector);
        expect(message.embedding?.vectorId).toBe(mockEmbedding.vectorId);
      });
    });

    describe('createMessageWithEmbeddingAndContext', () => {
      it('should create message with embedding and context in transaction', async () => {
        // Создаем чат и исходное сообщение для контекста
        const chat = await prisma.chat.create({
          data: mockChat
        });

        const sourceMessage = await prisma.message.create({
          data: {
            ...mockMessage,
            chatId: chat.id
          }
        });

        const message = await repository.createMessageWithEmbeddingAndContext(
          {
            ...mockMessage,
            chatId: chat.id
          },
          mockEmbedding,
          [{
            ...mockContext,
            sourceId: sourceMessage.id
          }]
        ) as MessageWithRelations;

        expect(message).toBeDefined();
        expect(message.chatId).toBe(chat.id);
        expect(message.message).toBe(mockMessage.message);
        expect(message.embedding).toBeDefined();
        expect(message.embedding?.vector).toEqual(mockEmbedding.vector);
        expect(message.embedding?.vectorId).toBe(mockEmbedding.vectorId);
        expect(message.usedContext).toBeDefined();
        expect(message.usedContext).toHaveLength(1);
        expect(message.usedContext?.[0].sourceId).toBe(sourceMessage.id);
        expect(message.usedContext?.[0].score).toBe(mockContext.score);
        expect(message.usedContext?.[0].usedInPrompt).toBe(mockContext.usedInPrompt);
      });

      it('should create message with embedding and no context if context array is empty', async () => {
        // Создаем чат для теста
        const chat = await prisma.chat.create({
          data: mockChat
        });

        const message = await repository.createMessageWithEmbeddingAndContext(
          {
            ...mockMessage,
            chatId: chat.id
          },
          mockEmbedding,
          []
        ) as MessageWithRelations;

        expect(message).toBeDefined();
        expect(message.chatId).toBe(chat.id);
        expect(message.message).toBe(mockMessage.message);
        expect(message.embedding).toBeDefined();
        expect(message.embedding?.vector).toEqual(mockEmbedding.vector);
        expect(message.embedding?.vectorId).toBe(mockEmbedding.vectorId);
        expect(message.usedContext).toHaveLength(0);
      });
    });
  });
});
