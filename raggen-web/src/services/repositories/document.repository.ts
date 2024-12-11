import { PrismaClient, Document } from '@prisma/client';
import { BaseRepository } from './base.repository';

export interface CreateDocumentParams {
  name: string;
  type: string;
  size: number;
  content: string;
  metadata?: string;
}

export class DocumentRepository extends BaseRepository {
  constructor(prisma: PrismaClient) {
    super(prisma);
  }

  /**
   * Создает новый документ
   */
  async create(params: CreateDocumentParams): Promise<Document> {
    return this.prisma.document.create({
      data: params
    });
  }

  /**
   * Получает документ по ID
   */
  async findById(id: string): Promise<Document | null> {
    return this.prisma.document.findUnique({
      where: { id }
    });
  }

  /**
   * Получает все документы
   */
  async findAll(): Promise<Document[]> {
    return this.prisma.document.findMany({
      orderBy: { createdAt: 'desc' }
    });
  }

  /**
   * Обновляет документ
   */
  async update(id: string, data: Partial<CreateDocumentParams>): Promise<Document> {
    return this.prisma.document.update({
      where: { id },
      data
    });
  }

  /**
   * Удаляет документ
   */
  async delete(id: string): Promise<Document> {
    return this.prisma.document.delete({
      where: { id }
    });
  }

  /**
   * Удаляет все документы (для тестов)
   */
  async deleteAll(): Promise<void> {
    await this.prisma.document.deleteMany();
  }
}
