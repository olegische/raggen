import { PrismaClient, Embedding, Context } from '@prisma/client';
import { BaseRepository } from './base.repository';

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

export class EmbeddingRepository extends BaseRepository {
  constructor(prisma: PrismaClient) {
    super(prisma);
  }

  // Embedding operations
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

  // Context operations
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
}
