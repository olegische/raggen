import { Document } from '@prisma/client';
import { JSDOM } from 'jsdom';
import { marked } from 'marked';
import { BaseRepository } from './repositories/base.repository';
import { DocumentRepository } from './repositories/document.repository';

export interface DocumentMetadata {
  author?: string;
  createdAt?: string;
  [key: string]: any;
}

export class DocumentService {
  private readonly supportedTypes = ['txt', 'md', 'html'];
  private readonly maxFileSize = 10 * 1024 * 1024; // 10MB
  private documentRepository: DocumentRepository;

  constructor(baseRepository?: BaseRepository) {
    const repo = baseRepository || new BaseRepository();
    this.documentRepository = repo.createRepository(DocumentRepository);
  }

  /**
   * Проверяет, поддерживается ли тип файла
   */
  isTypeSupported(type: string): boolean {
    return this.supportedTypes.includes(type.toLowerCase());
  }

  /**
   * Проверяет размер файла
   */
  isFileSizeValid(size: number): boolean {
    return size <= this.maxFileSize;
  }

  /**
   * Обрабатывает содержимое документа в зависимости от его типа
   */
  async processContent(content: string, type: string): Promise<string> {
    switch (type.toLowerCase()) {
      case 'html':
        return this.processHtmlContent(content);
      case 'md':
        return this.processMarkdownContent(content);
      case 'txt':
        return content;
      default:
        throw new Error(`Unsupported document type: ${type}`);
    }
  }

  /**
   * Обрабатывает HTML контент
   */
  private processHtmlContent(content: string): string {
    const dom = new JSDOM(content);
    const document = dom.window.document;
    
    // Удаляем скрипты и стили
    document.querySelectorAll('script, style').forEach(el => el.remove());
    
    // Получаем текст, сохраняя пробелы и переносы строк
    return document.body.textContent?.trim() || '';
  }

  /**
   * Обрабатывает Markdown контент
   */
  private processMarkdownContent(content: string): string {
    // Конвертируем markdown в HTML с опцией gfm для поддержки GitHub Flavored Markdown
    const html = marked.parse(content, { 
      gfm: true,
      breaks: true,
      async: false 
    }) as string;

    // Обрабатываем полученный HTML для извлечения чистого текста
    const dom = new JSDOM(html);
    const text = dom.window.document.body.textContent?.trim() || '';

    // Заменяем множественные пробелы и переносы строк на одиночные
    return text.replace(/\s+/g, ' ');
  }

  /**
   * Создает новый документ
   */
  async createDocument(
    name: string,
    type: string,
    content: string,
    size: number,
    metadata?: DocumentMetadata
  ): Promise<Document> {
    // Проверяем тип файла
    if (!this.isTypeSupported(type)) {
      throw new Error(`Unsupported document type: ${type}`);
    }

    // Проверяем размер файла
    if (!this.isFileSizeValid(size)) {
      throw new Error(`File size exceeds maximum allowed size of ${this.maxFileSize} bytes`);
    }

    try {
      // Обрабатываем содержимое
      const processedContent = await this.processContent(content, type);

      // Создаем документ
      return this.documentRepository.create({
        name,
        type: type.toLowerCase(),
        size,
        content: processedContent,
        metadata: metadata ? JSON.stringify(metadata) : undefined
      });
    } catch (error) {
      console.error('Error creating document:', error);
      throw error instanceof Error ? error : new Error('Failed to create document');
    }
  }

  /**
   * Получает документ по ID
   */
  async getDocument(id: string): Promise<Document | null> {
    try {
      return this.documentRepository.findById(id);
    } catch (error) {
      console.error('Error getting document:', error);
      throw error instanceof Error ? error : new Error('Failed to get document');
    }
  }

  /**
   * Получает все документы
   */
  async getAllDocuments(): Promise<Document[]> {
    try {
      return this.documentRepository.findAll();
    } catch (error) {
      console.error('Error getting all documents:', error);
      throw error instanceof Error ? error : new Error('Failed to get documents');
    }
  }

  /**
   * Обновляет документ
   */
  async updateDocument(
    id: string,
    updates: {
      name?: string;
      content?: string;
      metadata?: DocumentMetadata;
    }
  ): Promise<Document> {
    try {
      const document = await this.documentRepository.findById(id);
      if (!document) {
        throw new Error(`Document not found: ${id}`);
      }

      const updateData: any = {};

      if (updates.name) {
        updateData.name = updates.name;
      }

      if (updates.content) {
        updateData.content = await this.processContent(updates.content, document.type);
      }

      if (updates.metadata) {
        updateData.metadata = JSON.stringify(updates.metadata);
      }

      return this.documentRepository.update(id, updateData);
    } catch (error) {
      console.error('Error updating document:', error);
      if (error instanceof Error && error.message.includes('Document not found')) {
        throw error;
      }
      throw new Error('Failed to update document');
    }
  }

  /**
   * Удаляет документ
   */
  async deleteDocument(id: string): Promise<Document> {
    try {
      return this.documentRepository.delete(id);
    } catch (error) {
      console.error('Error deleting document:', error);
      throw error instanceof Error ? error : new Error('Failed to delete document');
    }
  }

  /**
   * Получает метаданные документа
   */
  getDocumentMetadata(document: Document): DocumentMetadata | null {
    if (!document.metadata) {
      return null;
    }

    try {
      return JSON.parse(document.metadata);
    } catch (error) {
      console.error('Error parsing document metadata:', error);
      return null;
    }
  }

  /**
   * Получает список поддерживаемых типов файлов
   */
  getSupportedTypes(): string[] {
    return [...this.supportedTypes];
  }

  /**
   * Получает максимальный размер файла
   */
  getMaxFileSize(): number {
    return this.maxFileSize;
  }
}
