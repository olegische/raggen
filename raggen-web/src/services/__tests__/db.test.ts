import { PrismaClient } from '@prisma/client';
import { mockDeep, mockReset, DeepMockProxy } from 'jest-mock-extended';

describe('Database Schema', () => {
  let prisma: DeepMockProxy<PrismaClient>;

  beforeEach(() => {
    prisma = mockDeep<PrismaClient>();
    mockReset(prisma);
  });

  describe('Chat Model', () => {
    const mockChat = {
      id: 'test-id',
      provider: 'yandex',
      createdAt: new Date(),
      updatedAt: new Date(),
      messages: []
    };

    it('should create chat with provider', async () => {
      prisma.chat.create.mockResolvedValue(mockChat);

      const chat = await prisma.chat.create({
        data: {
          provider: 'yandex'
        }
      });

      expect(chat.provider).toBe('yandex');
      expect(prisma.chat.create).toHaveBeenCalled();
    });

    it('should find chats by provider', async () => {
      prisma.chat.findMany.mockResolvedValue([mockChat]);

      const chats = await prisma.chat.findMany({
        where: {
          provider: 'yandex'
        }
      });

      expect(chats).toHaveLength(1);
      expect(chats[0].provider).toBe('yandex');
    });
  });

  describe('Message Model', () => {
    const mockMessage = {
      id: 1,
      chatId: 'test-chat-id',
      message: 'Test message',
      response: 'Test response',
      model: 'test-model',
      timestamp: new Date(),
      temperature: 0.7,
      maxTokens: 1000,
      provider: 'yandex'
    };

    it('should create message with model info', async () => {
      prisma.message.create.mockResolvedValue(mockMessage);

      const message = await prisma.message.create({
        data: {
          chatId: 'test-chat-id',
          message: 'Test message',
          model: 'test-model',
          temperature: 0.7,
          maxTokens: 1000,
          provider: 'yandex'
        }
      });

      expect(message.model).toBe('test-model');
      expect(message.temperature).toBe(0.7);
      expect(message.maxTokens).toBe(1000);
      expect(message.provider).toBe('yandex');
    });

    it('should find messages by chat id', async () => {
      prisma.message.findMany.mockResolvedValue([mockMessage]);

      const messages = await prisma.message.findMany({
        where: {
          chatId: 'test-chat-id'
        }
      });

      expect(messages).toHaveLength(1);
      expect(messages[0].chatId).toBe('test-chat-id');
      expect(messages[0].provider).toBe('yandex');
    });
  });
}); 