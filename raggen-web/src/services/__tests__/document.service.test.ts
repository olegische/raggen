import { Document } from '@prisma/client';
import { DocumentService } from '../document.service';
import { BaseRepository } from '../repositories/base.repository';
import '@testing-library/jest-dom';

describe('DocumentService', () => {
  let service: DocumentService;
  let baseRepository: BaseRepository;

  beforeEach(() => {
    baseRepository = new BaseRepository();
    service = new DocumentService(baseRepository);
  });

  describe('File validation', () => {
    it('should validate supported file types', () => {
      expect(service.isTypeSupported('txt')).toBe(true);
      expect(service.isTypeSupported('md')).toBe(true);
      expect(service.isTypeSupported('html')).toBe(true);
      expect(service.isTypeSupported('pdf')).toBe(false);
      expect(service.isTypeSupported('')).toBe(false);
    });

    it('should validate file size', () => {
      const maxSize = service.getMaxFileSize();
      expect(service.isFileSizeValid(maxSize)).toBe(true);
      expect(service.isFileSizeValid(maxSize - 1)).toBe(true);
      expect(service.isFileSizeValid(maxSize + 1)).toBe(false);
      expect(service.isFileSizeValid(0)).toBe(true);
    });

    it('should return supported types', () => {
      const types = service.getSupportedTypes();
      expect(types).toContain('txt');
      expect(types).toContain('md');
      expect(types).toContain('html');
      expect(types.length).toBe(3);
    });
  });

  describe('Content processing', () => {
    it('should process plain text content', async () => {
      const content = 'Hello, world!';
      const processed = await service.processContent(content, 'txt');
      expect(processed).toBe(content);
    });

    it('should process HTML content', async () => {
      const content = `
        <html>
          <head>
            <style>body { color: red; }</style>
          </head>
          <body>
            <script>alert('test');</script>
            <div>Hello, <span>world!</span></div>
          </body>
        </html>
      `;
      const processed = await service.processContent(content, 'html');
      expect(processed).toBe('Hello, world!');
    });

    it('should process Markdown content', async () => {
      const content = `
# Title

**Bold text**

- List item 1
- List item 2

[Link](https://example.com)
      `;
      const processed = await service.processContent(content, 'md');
      expect(processed).toBe('Title Bold text List item 1 List item 2 Link');
    });

    it('should throw error for unsupported type', async () => {
      await expect(service.processContent('test', 'pdf'))
        .rejects.toThrow('Unsupported document type: pdf');
    });
  });

  describe('Document operations', () => {
    const mockDocument = {
      name: 'test.txt',
      type: 'txt',
      content: 'Test content',
      size: 100,
      metadata: {
        author: 'Test Author',
        createdAt: new Date().toISOString()
      }
    };

    describe('createDocument', () => {
      it('should create document with valid params', async () => {
        const document = await service.createDocument(
          mockDocument.name,
          mockDocument.type,
          mockDocument.content,
          mockDocument.size,
          mockDocument.metadata
        );

        expect(document).toBeDefined();
        expect(document.name).toBe(mockDocument.name);
        expect(document.type).toBe(mockDocument.type);
        expect(document.content).toBe(mockDocument.content);
        expect(document.size).toBe(mockDocument.size);

        const metadata = service.getDocumentMetadata(document);
        expect(metadata).toBeDefined();
        expect(metadata?.author).toBe(mockDocument.metadata.author);
        expect(metadata?.createdAt).toBe(mockDocument.metadata.createdAt);
      });

      it('should throw error for unsupported type', async () => {
        await expect(service.createDocument(
          'test.pdf',
          'pdf',
          'content',
          100
        )).rejects.toThrow('Unsupported document type: pdf');
      });

      it('should throw error for invalid size', async () => {
        const maxSize = service.getMaxFileSize();
        await expect(service.createDocument(
          'test.txt',
          'txt',
          'content',
          maxSize + 1
        )).rejects.toThrow('File size exceeds maximum allowed size');
      });
    });

    describe('getDocument', () => {
      it('should get document by id', async () => {
        const created = await service.createDocument(
          mockDocument.name,
          mockDocument.type,
          mockDocument.content,
          mockDocument.size
        );

        const document = await service.getDocument(created.id);
        expect(document).toBeDefined();
        expect(document?.id).toBe(created.id);
        expect(document?.name).toBe(mockDocument.name);
      });

      it('should return null for non-existent document', async () => {
        const document = await service.getDocument('non-existent-id');
        expect(document).toBeNull();
      });
    });

    describe('getAllDocuments', () => {
      it('should get all documents', async () => {
        await service.createDocument(
          'doc1.txt',
          'txt',
          'content 1',
          100
        );

        await service.createDocument(
          'doc2.txt',
          'txt',
          'content 2',
          100
        );

        const documents = await service.getAllDocuments();
        expect(documents.length).toBeGreaterThanOrEqual(2);
        expect(documents.some(d => d.name === 'doc1.txt')).toBe(true);
        expect(documents.some(d => d.name === 'doc2.txt')).toBe(true);
      });
    });

    describe('updateDocument', () => {
      it('should update document with valid params', async () => {
        const created = await service.createDocument(
          mockDocument.name,
          mockDocument.type,
          mockDocument.content,
          mockDocument.size
        );

        const updates = {
          name: 'updated.txt',
          content: 'Updated content',
          metadata: { author: 'New Author' }
        };

        const updated = await service.updateDocument(created.id, updates);
        expect(updated.id).toBe(created.id);
        expect(updated.name).toBe(updates.name);
        expect(updated.content).toBe(updates.content);

        const metadata = service.getDocumentMetadata(updated);
        expect(metadata?.author).toBe('New Author');
      });

      it('should throw error for non-existent document', async () => {
        await expect(service.updateDocument('non-existent-id', {
          name: 'updated.txt'
        })).rejects.toThrow('Document not found');
      });
    });

    describe('deleteDocument', () => {
      it('should delete document', async () => {
        const created = await service.createDocument(
          mockDocument.name,
          mockDocument.type,
          mockDocument.content,
          mockDocument.size
        );

        const deleted = await service.deleteDocument(created.id);
        expect(deleted.id).toBe(created.id);

        const found = await service.getDocument(created.id);
        expect(found).toBeNull();
      });
    });

    describe('metadata handling', () => {
      it('should handle document metadata', async () => {
        const created = await service.createDocument(
          mockDocument.name,
          mockDocument.type,
          mockDocument.content,
          mockDocument.size,
          mockDocument.metadata
        );

        const metadata = service.getDocumentMetadata(created);
        expect(metadata).toBeDefined();
        expect(metadata?.author).toBe(mockDocument.metadata.author);
        expect(metadata?.createdAt).toBe(mockDocument.metadata.createdAt);
      });

      it('should return null for missing metadata', () => {
        const document = {
          id: '1',
          name: 'test.txt',
          type: 'txt',
          content: 'test',
          size: 100,
          metadata: null,
          createdAt: new Date(),
          updatedAt: new Date()
        } as Document;

        const metadata = service.getDocumentMetadata(document);
        expect(metadata).toBeNull();
      });

      it('should handle invalid metadata JSON', () => {
        const document = {
          id: '1',
          name: 'test.txt',
          type: 'txt',
          content: 'test',
          size: 100,
          metadata: 'invalid json',
          createdAt: new Date(),
          updatedAt: new Date()
        } as Document;

        const metadata = service.getDocumentMetadata(document);
        expect(metadata).toBeNull();
      });
    });
  });
});
