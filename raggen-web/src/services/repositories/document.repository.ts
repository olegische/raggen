import { PrismaClient, Document } from '@prisma/client';
import { BaseRepository } from './base.repository';

export interface CreateDocumentParams {
  name: string;
  type: string;
  size: number;
  content: string;
  metadata?: string;
}

export interface UpdateDocumentParams {
  name?: string;
  type?: string;
  size?: number;
  content?: string;
  metadata?: string;
}

export class DocumentRepository extends BaseRepository {
  constructor(prisma: PrismaClient) {
    super(prisma);
  }

  async create(params: CreateDocumentParams): Promise<Document> {
    try {
      this.validateCreateParams(params);

      return await this.prisma.document.create({
        data: params
      });
    } catch (error) {
      console.error('Error creating document:', error);
      throw error instanceof Error ? error : new Error('Failed to create document');
    }
  }

  async findById(id: string): Promise<Document | null> {
    try {
      if (!id || id.trim().length === 0) {
        throw new Error('Document ID is required');
      }

      return await this.prisma.document.findUnique({
        where: { id }
      });
    } catch (error) {
      console.error('Error getting document:', error);
      throw error instanceof Error ? error : new Error('Failed to get document');
    }
  }

  async findByType(type: string): Promise<Document[]> {
    try {
      if (!type || type.trim().length === 0) {
        throw new Error('Document type is required');
      }

      return await this.prisma.document.findMany({
        where: { type },
        orderBy: { createdAt: 'desc' }
      });
    } catch (error) {
      console.error('Error getting documents by type:', error);
      throw error instanceof Error ? error : new Error('Failed to get documents');
    }
  }

  async findAll(): Promise<Document[]> {
    try {
      return await this.prisma.document.findMany({
        orderBy: { createdAt: 'desc' }
      });
    } catch (error) {
      console.error('Error getting all documents:', error);
      throw error instanceof Error ? error : new Error('Failed to get documents');
    }
  }

  async update(id: string, params: UpdateDocumentParams): Promise<Document> {
    try {
      if (!id || id.trim().length === 0) {
        throw new Error('Document ID is required');
      }

      this.validateUpdateParams(params);

      return await this.prisma.document.update({
        where: { id },
        data: params
      });
    } catch (error) {
      console.error('Error updating document:', error);
      throw error instanceof Error ? error : new Error('Failed to update document');
    }
  }

  async delete(id: string): Promise<Document> {
    try {
      if (!id || id.trim().length === 0) {
        throw new Error('Document ID is required');
      }

      return await this.prisma.document.delete({
        where: { id }
      });
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error instanceof Error ? error : new Error('Failed to delete document');
    }
  }

  async deleteAll(): Promise<void> {
    try {
      await this.prisma.document.deleteMany({});
    } catch (error) {
      console.error('Error deleting all documents:', error);
      throw error instanceof Error ? error : new Error('Failed to delete all documents');
    }
  }

  private validateCreateParams(params: CreateDocumentParams): void {
    if (!params.name || params.name.trim().length === 0) {
      throw new Error('Document name is required');
    }

    if (!params.type || params.type.trim().length === 0) {
      throw new Error('Document type is required');
    }

    if (!params.content || params.content.trim().length === 0) {
      throw new Error('Document content is required');
    }

    if (typeof params.size !== 'number' || params.size <= 0) {
      throw new Error('Document size must be greater than 0');
    }

    const allowedTypes = ['txt', 'md', 'html'];
    if (!allowedTypes.includes(params.type.toLowerCase())) {
      throw new Error(`Document type must be one of: ${allowedTypes.join(', ')}`);
    }

    if (params.metadata !== undefined) {
      try {
        JSON.parse(params.metadata);
      } catch (error) {
        throw new Error('Invalid metadata JSON format');
      }
    }
  }

  private validateUpdateParams(params: UpdateDocumentParams): void {
    if (params.name !== undefined && params.name.trim().length === 0) {
      throw new Error('Document name cannot be empty');
    }

    if (params.type !== undefined) {
      if (params.type.trim().length === 0) {
        throw new Error('Document type cannot be empty');
      }

      const allowedTypes = ['txt', 'md', 'html'];
      if (!allowedTypes.includes(params.type.toLowerCase())) {
        throw new Error(`Document type must be one of: ${allowedTypes.join(', ')}`);
      }
    }

    if (params.content !== undefined && params.content.trim().length === 0) {
      throw new Error('Document content cannot be empty');
    }

    if (params.size !== undefined && (typeof params.size !== 'number' || params.size <= 0)) {
      throw new Error('Document size must be greater than 0');
    }

    if (params.metadata !== undefined) {
      try {
        JSON.parse(params.metadata);
      } catch (error) {
        throw new Error('Invalid metadata JSON format');
      }
    }
  }
}
