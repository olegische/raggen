import { PrismaClient, Message } from '@prisma/client';
import { MessageRepository } from '../message.repository';
import { BaseRepository } from '../base.repository';
import '@testing-library/jest-dom';

describe('MessageRepository', () => {
  let repository: MessageRepository;
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
    repository = baseRepository.createRepository(MessageRepository);
  });

  afterEach(async () => {
    // Очищаем тестовые данные
    await prisma.message.deleteMany();
    await prisma.chat.deleteMany();
  });

  describe('Message operations', () => {
    const mockChat = {
      id: 'test-chat-id',
      provider: 'test'
    };

    const mockMessage = {
      chatId: 'test-chat-id',
      message: 'Test message',
      model: 'test-model',
      provider: 'test-provider',
      temperature: 0.7,
      maxTokens: 1000,
      response: null
    };

    describe('createMessage', () => {
      it('should create message with valid params', async () => {
        // Создаем чат для теста
        await prisma.chat.create({
          data: mockChat
        });

        const message = await repository.createMessage(mockMessage);

        expect(message).toBeDefined();
        expect(message.chatId).toBe(mockMessage.chatId);
        expect(message.message).toBe(mockMessage.message);
        expect(message.model).toBe(mockMessage.model);
        expect(message.provider).toBe(mockMessage.provider);
        expect(message.temperature).toBe(mockMessage.temperature);
        expect(message.maxTokens).toBe(mockMessage.maxTokens);
        expect(message.response).toBe(mockMessage.response);
        expect(message.id).toBeDefined();
        expect(message.timestamp).toBeDefined();
      });

      it('should throw error for missing chat ID', async () => {
        await expect(repository.createMessage({
          ...mockMessage,
          chatId: ''
        })).rejects.toThrow('Chat ID is required');
      });

      it('should throw error for empty message', async () => {
        await expect(repository.createMessage({
          ...mockMessage,
          message: ''
        })).rejects.toThrow('Message content is required');
      });

      it('should throw error for empty model', async () => {
        await expect(repository.createMessage({
          ...mockMessage,
          model: ''
        })).rejects.toThrow('Model is required');
      });

      it('should throw error for empty provider', async () => {
        await expect(repository.createMessage({
          ...mockMessage,
          provider: ''
        })).rejects.toThrow('Provider is required');
      });

      it('should throw error for invalid temperature', async () => {
        await expect(repository.createMessage({
          ...mockMessage,
          temperature: 1.5
        })).rejects.toThrow('Temperature must be between 0 and 1');
      });

      it('should throw error for invalid maxTokens', async () => {
        await expect(repository.createMessage({
          ...mockMessage,
          maxTokens: 0
        })).rejects.toThrow('Max tokens must be greater than 0');
      });
    });

    describe('getMessage', () => {
      it('should get message by valid id', async () => {
        // Создаем чат и сообщение для теста
        await prisma.chat.create({
          data: mockChat
        });

        const createdMessage = await prisma.message.create({
          data: mockMessage
        });

        const message = await repository.getMessage(createdMessage.id);

        expect(message).toBeDefined();
        expect(message?.id).toBe(createdMessage.id);
        expect(message?.chatId).toBe(mockMessage.chatId);
        expect(message?.message).toBe(mockMessage.message);
      });

      it('should throw error for invalid message ID', async () => {
        await expect(repository.getMessage(0))
          .rejects.toThrow('Invalid message ID');
      });

      it('should return null when message not found', async () => {
        const message = await repository.getMessage(999);
        expect(message).toBeNull();
      });
    });

    describe('getMessagesByChat', () => {
      it('should get messages by valid chat id', async () => {
        // Создаем чат и сообщения для теста
        await prisma.chat.create({
          data: mockChat
        });

        const msg1 = await prisma.message.create({
          data: {
            ...mockMessage,
            message: 'Message 1'
          }
        });

        const msg2 = await prisma.message.create({
          data: {
            ...mockMessage,
            message: 'Message 2'
          }
        });

        const messages = await repository.getMessagesByChat(mockChat.id);

        expect(messages).toHaveLength(2);
        expect(messages.map(m => m.id)).toContain(msg1.id);
        expect(messages.map(m => m.id)).toContain(msg2.id);
        // Проверяем сортировку по timestamp asc
        expect(messages[0].timestamp.getTime()).toBeLessThanOrEqual(messages[1].timestamp.getTime());
      });

      it('should throw error for invalid chat ID', async () => {
        await expect(repository.getMessagesByChat(''))
          .rejects.toThrow('Invalid chat ID');
      });

      it('should return empty array when no messages found', async () => {
        const messages = await repository.getMessagesByChat('non-existent-chat');
        expect(messages).toHaveLength(0);
      });
    });

    describe('updateMessage', () => {
      it('should update message with valid params', async () => {
        // Создаем чат и сообщение для теста
        await prisma.chat.create({
          data: mockChat
        });

        const message = await prisma.message.create({
          data: mockMessage
        });

        const updatedMessage = await repository.updateMessage(message.id, {
          response: 'Test response'
        });

        expect(updatedMessage).toBeDefined();
        expect(updatedMessage.id).toBe(message.id);
        expect(updatedMessage.response).toBe('Test response');
      });

      it('should throw error for invalid message ID', async () => {
        await expect(repository.updateMessage(0, {
          response: 'Test response'
        })).rejects.toThrow('Invalid message ID');
      });
    });

    describe('deleteMessage', () => {
      it('should delete message', async () => {
        // Создаем чат и сообщение для теста
        await prisma.chat.create({
          data: mockChat
        });

        const message = await prisma.message.create({
          data: mockMessage
        });

        const deletedMessage = await repository.deleteMessage(message.id);
        expect(deletedMessage.id).toBe(message.id);

        const found = await prisma.message.findUnique({
          where: { id: message.id }
        });
        expect(found).toBeNull();
      });

      it('should throw error for invalid message ID', async () => {
        await expect(repository.deleteMessage(0))
          .rejects.toThrow('Invalid message ID');
      });
    });
  });
});
