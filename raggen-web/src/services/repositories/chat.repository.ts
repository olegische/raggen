import { PrismaClient, Chat, Message } from '@prisma/client';
import { BaseRepository } from './base.repository';

export interface CreateChatParams {
  provider: string;
}

export interface CreateMessageParams {
  chatId: string;
  message: string;
  model: string;
  provider: string;
  temperature: number;
  maxTokens: number;
  response?: string | null;
}

export class ChatRepository extends BaseRepository {
  constructor(prisma: PrismaClient) {
    super(prisma);
  }

  // Chat operations
  async createChat(params: CreateChatParams): Promise<Chat> {
    return this.prisma.chat.create({
      data: params
    });
  }

  async getChat(id: string): Promise<Chat | null> {
    return this.prisma.chat.findUnique({
      where: { id }
    });
  }

  async getChatsByProvider(provider: string): Promise<Chat[]> {
    return this.prisma.chat.findMany({
      where: { provider },
      orderBy: { createdAt: 'desc' }
    });
  }

  // Message operations
  async createMessage(params: CreateMessageParams): Promise<Message> {
    return this.prisma.message.create({
      data: params
    });
  }

  async getMessage(id: number): Promise<Message | null> {
    return this.prisma.message.findUnique({
      where: { id },
      include: {
        embedding: true,
        usedContext: true
      }
    });
  }

  async getMessagesByChat(chatId: string): Promise<Message[]> {
    return this.prisma.message.findMany({
      where: { chatId },
      include: {
        embedding: true,
        usedContext: true
      },
      orderBy: { timestamp: 'asc' }
    });
  }

  async getMessagesByIds(ids: number[]): Promise<Message[]> {
    return this.prisma.message.findMany({
      where: {
        id: {
          in: ids
        }
      }
    });
  }

  async getMessagesWithoutEmbeddings(): Promise<Message[]> {
    return this.prisma.message.findMany({
      where: {
        embedding: null
      }
    });
  }

  // Transaction operations
  async createMessageWithEmbedding(
    messageParams: CreateMessageParams,
    embeddingParams: { vector: Buffer; vectorId: number }
  ): Promise<Message> {
    return this.prisma.$transaction(async (tx) => {
      // Создаем сообщение
      const message = await tx.message.create({
        data: messageParams
      });

      // Создаем эмбеддинг
      await tx.embedding.create({
        data: {
          messageId: message.id,
          vector: embeddingParams.vector,
          vectorId: embeddingParams.vectorId
        }
      });

      return message;
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
        data: messageParams
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

      return message;
    });
  }
}
