import { PrismaClient, Embedding, Context, Message, Document } from '@prisma/client';
import { BaseRepository } from './base.repository';
import { MessageRepository } from './message.repository';
import { DocumentRepository } from './document.repository';

export interface CreateEmbeddingParams {
  messageId?: number;
  documentId?: string;
  vector: Buffer;
  vectorId: number;
}

export interface CreateContextParams {
  messageId: number;
  sourceId: number;
  documentId?: string;
  score: number;
  usedInPrompt: boolean;
}

export type EmbeddingWithRelations = Embedding & {
  message: Message | null;
  document: Document | null;
};

export type ContextWithRelations = Context & {
  message: Message;
  document: Document | null;
};

export class EmbeddingRepository extends BaseRepository {
  private messageRepository: MessageRepository;
  private documentRepository: DocumentRepository;

  constructor(prisma: PrismaClient) {
    super(prisma);
    this.messageRepository = new MessageRepository(prisma);
    this.documentRepository = new DocumentRepository(prisma);
  }

  // Embedding operations
  async createEmbedding(params: CreateEmbeddingParams): Promise<Embedding> {
    try {
      this.validateEmbeddingParams(params);

      // Проверяем существование сообщения или документа
      if (params.messageId) {
        const message = await this.messageRepository.getMessage(params.messageId);
        if (!message) {
          throw new Error('Message not found');
        }
      } else if (params.documentId) {
        const document = await this.documentRepository.findById(params.documentId);
        if (!document) {
          throw new Error('Document not found');
        }
      }

      return await this.prisma.embedding.create({
        data: params
      });
    } catch (error) {
      console.error('Error creating embedding:', error);
      throw error instanceof Error ? error : new Error('Failed to create embedding');
    }
  }

  async getEmbeddingByMessageId(messageId: number): Promise<Embedding | null> {
    try {
      if (!messageId || messageId <= 0) {
        throw new Error('Invalid message ID');
      }

      // Проверяем существование сообщения
      const message = await this.messageRepository.getMessage(messageId);
      if (!message) {
        return null;
      }

      return await this.prisma.embedding.findUnique({
        where: { messageId }
      });
    } catch (error) {
      console.error('Error getting embedding by message ID:', error);
      throw error instanceof Error ? error : new Error('Failed to get embedding');
    }
  }

  async getEmbeddingByDocumentId(documentId: string): Promise<Embedding | null> {
    try {
      if (!documentId) {
        throw new Error('Invalid document ID');
      }

      // Проверяем существование документа
      const document = await this.documentRepository.findById(documentId);
      if (!document) {
        return null;
      }

      return await this.prisma.embedding.findUnique({
        where: { documentId }
      });
    } catch (error) {
      console.error('Error getting embedding by document ID:', error);
      throw error instanceof Error ? error : new Error('Failed to get embedding');
    }
  }

  async getEmbeddingsByVectorIds(vectorIds: number[]): Promise<EmbeddingWithRelations[]> {
    try {
      if (!vectorIds.length) {
        throw new Error('Vector IDs array is empty');
      }

      if (vectorIds.some(id => id <= 0)) {
        throw new Error('Invalid vector ID in array');
      }

      const embeddings = await this.prisma.embedding.findMany({
        where: {
          vectorId: {
            in: vectorIds
          }
        },
        include: {
          message: true,
          document: true
        }
      });

      // Filter out embeddings without both message and document
      return embeddings.filter((embedding): embedding is EmbeddingWithRelations => 
        embedding.message !== null || embedding.document !== null
      );
    } catch (error) {
      console.error('Error getting embeddings by vector IDs:', error);
      throw error instanceof Error ? error : new Error('Failed to get embeddings');
    }
  }

  // Context operations
  async createContext(params: CreateContextParams): Promise<Context> {
    try {
      this.validateContextParams(params);

      // Проверяем существование сообщений и документа
      const message = await this.messageRepository.getMessage(params.messageId);
      if (!message) {
        throw new Error('Message not found');
      }

      const sourceMessage = await this.messageRepository.getMessage(params.sourceId);
      if (!sourceMessage) {
        throw new Error('Source message not found');
      }

      if (params.documentId) {
        const document = await this.documentRepository.findById(params.documentId);
        if (!document) {
          throw new Error('Document not found');
        }
      }

      return await this.prisma.context.create({
        data: params
      });
    } catch (error) {
      console.error('Error creating context:', error);
      throw error instanceof Error ? error : new Error('Failed to create context');
    }
  }

  async createManyContexts(contexts: CreateContextParams[]): Promise<void> {
    try {
      if (!contexts.length) {
        throw new Error('Contexts array is empty');
      }

      // Проверяем валидность всех контекстов
      contexts.forEach(this.validateContextParams);

      // Проверяем существование всех сообщений
      const messageIds = contexts.map(c => c.messageId);
      const sourceIds = contexts.map(c => c.sourceId);
      const allIds = Array.from(new Set([...messageIds, ...sourceIds]));

      const messages = await Promise.all(
        allIds.map(id => this.messageRepository.getMessage(id))
      );

      const missingIds = messages.some(m => m === null);
      if (missingIds) {
        throw new Error('Some messages not found');
      }

      // Проверяем существование всех документов
      const documentIds = contexts
        .map(c => c.documentId)
        .filter((id): id is string => id !== undefined);

      if (documentIds.length > 0) {
        const documents = await Promise.all(
          documentIds.map(id => this.documentRepository.findById(id))
        );

        const missingDocuments = documents.some(doc => doc === null);
        if (missingDocuments) {
          throw new Error('Some documents not found');
        }
      }

      await this.prisma.context.createMany({
        data: contexts
      });
    } catch (error) {
      console.error('Error creating multiple contexts:', error);
      throw error instanceof Error ? error : new Error('Failed to create contexts');
    }
  }

  async getContextByMessageId(messageId: number): Promise<ContextWithRelations[]> {
    try {
      if (!messageId || messageId <= 0) {
        throw new Error('Invalid message ID');
      }

      // Проверяем существование сообщения
      const message = await this.messageRepository.getMessage(messageId);
      if (!message) {
        return [];
      }

      const contexts = await this.prisma.context.findMany({
        where: { messageId },
        include: {
          message: true,
          document: true
        },
        orderBy: { score: 'desc' }
      });

      // Filter out contexts without message
      return contexts.filter((context): context is ContextWithRelations => 
        context.message !== null
      );
    } catch (error) {
      console.error('Error getting context by message ID:', error);
      throw error instanceof Error ? error : new Error('Failed to get context');
    }
  }

  async getContextByDocumentId(documentId: string): Promise<ContextWithRelations[]> {
    try {
      if (!documentId) {
        throw new Error('Invalid document ID');
      }

      // Проверяем существование документа
      const document = await this.documentRepository.findById(documentId);
      if (!document) {
        return [];
      }

      const contexts = await this.prisma.context.findMany({
        where: { documentId },
        include: {
          message: true,
          document: true
        },
        orderBy: { score: 'desc' }
      });

      // Filter out contexts without message
      return contexts.filter((context): context is ContextWithRelations => 
        context.message !== null
      );
    } catch (error) {
      console.error('Error getting context by document ID:', error);
      throw error instanceof Error ? error : new Error('Failed to get context');
    }
  }

  // Validation methods
  private validateEmbeddingParams(params: CreateEmbeddingParams): void {
    if (!params.messageId && !params.documentId) {
      throw new Error('Either message ID or document ID is required');
    }

    if (params.messageId && params.documentId) {
      throw new Error('Cannot create embedding for both message and document');
    }

    if (params.messageId && params.messageId <= 0) {
      throw new Error('Invalid message ID');
    }

    if (!params.vector || params.vector.length === 0) {
      throw new Error('Vector is required');
    }

    if (!params.vectorId || params.vectorId <= 0) {
      throw new Error('Invalid vector ID');
    }
  }

  private validateContextParams(params: CreateContextParams): void {
    if (!params.messageId || params.messageId <= 0) {
      throw new Error('Invalid message ID');
    }

    if (!params.sourceId || params.sourceId <= 0) {
      throw new Error('Invalid source ID');
    }

    if (typeof params.score !== 'number' || params.score < 0 || params.score > 1) {
      throw new Error('Score must be between 0 and 1');
    }

    if (typeof params.usedInPrompt !== 'boolean') {
      throw new Error('usedInPrompt must be a boolean');
    }
  }
}
