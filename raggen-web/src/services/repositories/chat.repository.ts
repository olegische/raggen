import { PrismaClient, Chat, Message } from '@prisma/client';
import { BaseRepository } from './base.repository';
import { MessageRepository, CreateMessageParams } from './message.repository';
import { EmbeddingRepository, CreateEmbeddingParams, CreateContextParams } from './embedding.repository';

export interface CreateChatParams {
  provider: string;
}

export class ChatRepository extends BaseRepository {
  private messageRepository: MessageRepository;
  private embeddingRepository: EmbeddingRepository;

  constructor(prisma: PrismaClient) {
    super(prisma);
    this.messageRepository = new MessageRepository(prisma);
    this.embeddingRepository = new EmbeddingRepository(prisma);
  }

  // Chat operations
  async createChat(params: CreateChatParams): Promise<Chat> {
    try {
      if (!params.provider || params.provider.trim().length === 0) {
        throw new Error('Provider is required');
      }

      return await this.prisma.chat.create({
        data: params
      });
    } catch (error) {
      console.error('Error creating chat:', error);
      throw error instanceof Error ? error : new Error('Failed to create chat');
    }
  }

  async getChat(id: string): Promise<Chat | null> {
    try {
      if (!id || id.trim().length === 0) {
        throw new Error('Chat ID is required');
      }

      return await this.prisma.chat.findUnique({
        where: { id }
      });
    } catch (error) {
      console.error('Error getting chat:', error);
      throw error instanceof Error ? error : new Error('Failed to get chat');
    }
  }

  async getChatsByProvider(provider: string): Promise<Chat[]> {
    try {
      if (!provider || provider.trim().length === 0) {
        throw new Error('Provider is required');
      }

      return await this.prisma.chat.findMany({
        where: { provider },
        orderBy: { createdAt: 'desc' }
      });
    } catch (error) {
      console.error('Error getting chats by provider:', error);
      throw error instanceof Error ? error : new Error('Failed to get chats');
    }
  }

  // Transaction operations
  async createMessageWithEmbedding(
    messageParams: CreateMessageParams,
    embeddingParams: { vector: Buffer; vectorId: number }
  ): Promise<Message> {
    return this.prisma.$transaction(async (tx) => {
      // Создаем сообщение
      const message = await tx.message.create({
        data: messageParams,
        include: {
          embedding: true,
          usedContext: true
        }
      });

      // Создаем эмбеддинг
      await tx.embedding.create({
        data: {
          messageId: message.id,
          vector: embeddingParams.vector,
          vectorId: embeddingParams.vectorId
        }
      });

      // Возвращаем сообщение с обновленными связями
      return tx.message.findUnique({
        where: { id: message.id },
        include: {
          embedding: true,
          usedContext: true
        }
      }) as Promise<Message>;
    });
  }

  async createMessageWithEmbeddingAndContext(
    messageParams: CreateMessageParams,
    embeddingParams: { vector: Buffer; vectorId: number },
    contextParams: { sourceId: number; score: number; usedInPrompt: boolean }[]
  ): Promise<Message> {
    return this.prisma.$transaction(async (tx) => {
      // Создаем сообщение
      const message = await tx.message.create({
        data: messageParams,
        include: {
          embedding: true,
          usedContext: true
        }
      });

      // Создаем эмбеддинг
      await tx.embedding.create({
        data: {
          messageId: message.id,
          vector: embeddingParams.vector,
          vectorId: embeddingParams.vectorId
        }
      });

      // Создаем контексты
      if (contextParams.length > 0) {
        await tx.context.createMany({
          data: contextParams.map(context => ({
            messageId: message.id,
            sourceId: context.sourceId,
            score: context.score,
            usedInPrompt: context.usedInPrompt
          }))
        });
      }

      // Возвращаем сообщение с обновленными связями
      return tx.message.findUnique({
        where: { id: message.id },
        include: {
          embedding: true,
          usedContext: true
        }
      }) as Promise<Message>;
    });
  }
}
