import { PrismaClient, Message, Embedding, Context } from '@prisma/client';

const prisma = new PrismaClient();

export interface CreateMessageParams {
  chatId: string;
  message: string;
  model: string;
  provider: string;
  temperature: number;
  maxTokens: number;
}

export interface CreateEmbeddingParams {
  messageId: number;
  vector: Buffer;
  vectorId: number;
}

export interface CreateContextParams {
  messageId: number;
  sourceId: number;
  score: number;
  usedInPrompt: boolean;
}

export class DatabaseService {
  // Сообщения
  async createMessage(params: CreateMessageParams): Promise<Message> {
    return prisma.message.create({
      data: params
    });
  }

  async getMessage(id: number): Promise<Message | null> {
    return prisma.message.findUnique({
      where: { id },
      include: {
        embedding: true,
        usedContext: true
      }
    });
  }

  async getMessagesByChat(chatId: string): Promise<Message[]> {
    return prisma.message.findMany({
      where: { chatId },
      include: {
        embedding: true,
        usedContext: true
      },
      orderBy: { timestamp: 'asc' }
    });
  }

  // Эмбеддинги
  async createEmbedding(params: CreateEmbeddingParams): Promise<Embedding> {
    return prisma.embedding.create({
      data: params
    });
  }

  async getEmbeddingByMessageId(messageId: number): Promise<Embedding | null> {
    return prisma.embedding.findUnique({
      where: { messageId }
    });
  }

  async getEmbeddingsByVectorIds(vectorIds: number[]): Promise<Embedding[]> {
    return prisma.embedding.findMany({
      where: {
        vectorId: {
          in: vectorIds
        }
      },
      include: {
        message: true
      }
    });
  }

  // Контекст
  async createContext(params: CreateContextParams): Promise<Context> {
    return prisma.context.create({
      data: params
    });
  }

  async createManyContexts(contexts: CreateContextParams[]): Promise<void> {
    await prisma.context.createMany({
      data: contexts
    });
  }

  async getContextByMessageId(messageId: number): Promise<Context[]> {
    return prisma.context.findMany({
      where: { messageId },
      include: {
        message: true
      },
      orderBy: { score: 'desc' }
    });
  }

  // Транзакции
  async createMessageWithEmbedding(
    messageParams: CreateMessageParams,
    embeddingParams: Omit<CreateEmbeddingParams, 'messageId'>
  ): Promise<Message> {
    return prisma.$transaction(async (tx) => {
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
    embeddingParams: Omit<CreateEmbeddingParams, 'messageId'>,
    contextParams: Omit<CreateContextParams, 'messageId'>[]
  ): Promise<Message> {
    return prisma.$transaction(async (tx) => {
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