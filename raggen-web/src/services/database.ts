import { PrismaClient, Message, Embedding, Context, Chat } from '@prisma/client';

export interface CreateMessageParams {
  chatId: string;
  message: string;
  model: string;
  provider: string;
  temperature: number;
  maxTokens: number;
  response?: string | null;
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

export interface CreateChatParams {
  provider: string;
}

export class DatabaseService {
  private prisma: PrismaClient;

  constructor(prismaClient?: PrismaClient) {
    this.prisma = prismaClient || new PrismaClient();
  }

  // Чаты
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

  // Сообщения
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

  // Эмбеддинги
  async createEmbedding(params: CreateEmbeddingParams): Promise<Embedding> {
    return this.prisma.embedding.create({
      data: params
    });
  }

  async getEmbeddingByMessageId(messageId: number): Promise<Embedding | null> {
    return this.prisma.embedding.findUnique({
      where: { messageId }
    });
  }

  async getEmbeddingsByVectorIds(vectorIds: number[]): Promise<Embedding[]> {
    return this.prisma.embedding.findMany({
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
    return this.prisma.context.create({
      data: params
    });
  }

  async createManyContexts(contexts: CreateContextParams[]): Promise<void> {
    await this.prisma.context.createMany({
      data: contexts
    });
  }

  async getContextByMessageId(messageId: number): Promise<Context[]> {
    return this.prisma.context.findMany({
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
    embeddingParams: Omit<CreateEmbeddingParams, 'messageId'>,
    contextParams: Omit<CreateContextParams, 'messageId'>[]
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

  /**
   * Получение сообщений по их идентификаторам
   */
  async getMessagesByIds(ids: number[]): Promise<Message[]> {
    return this.prisma.message.findMany({
      where: {
        id: {
          in: ids
        }
      }
    });
  }

  /**
   * Получение сообщений без эмбеддингов
   */
  async getMessagesWithoutEmbeddings(): Promise<Message[]> {
    return this.prisma.message.findMany({
      where: {
        embedding: null
      }
    });
  }
} 