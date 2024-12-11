import { PrismaClient, Document } from '@prisma/client';
import { DocumentRepository } from '../document.repository';
import { BaseRepository } from '../base.repository';
import '@testing-library/jest-dom';

describe('DocumentRepository', () => {
  let repository: DocumentRepository;
  let baseRepository: BaseRepository;
  let prisma: PrismaClient;

  beforeAll(async () => {
    prisma = new PrismaClient();
    await prisma.$connect();
  });

  afterAll(async () => {
    await prisma.$disconnect();
  });

  beforeEach(() => {
    baseRepository = new BaseRepository(prisma);
    repository = baseRepository.createRepository(DocumentRepository);
  });

  afterEach(async () => {
    await prisma.document.deleteMany();
  });

  describe('Document operations', () => {
    const mockDocument = {
      name: 'test.txt',
      type: 'txt',
      size: 100,
      content: 'Test content'
    };

    describe('create', () => {
      it('should create document with valid params', async () => {
        const document = await repository.create(mockDocument);

        expect(document).toBeDefined();
        expect(document.name).toBe(mockDocument.name);
        expect(document.type).toBe(mockDocument.type);
        expect(document.size).toBe(mockDocument.size);
        expect(document.content).toBe(mockDocument.content);
        expect(document.id).toBeDefined();
        expect(document.createdAt).toBeDefined();
        expect(document.updatedAt).toBeDefined();
      });

      it('should throw error for empty name', async () => {
        await expect(repository.create({
          ...mockDocument,
          name: ''
        })).rejects.toThrow('Document name is required');
      });

      it('should throw error for empty type', async () => {
        await expect(repository.create({
          ...mockDocument,
          type: ''
        })).rejects.toThrow('Document type is required');
      });

      it('should throw error for invalid type', async () => {
        await expect(repository.create({
          ...mockDocument,
          type: 'pdf'
        })).rejects.toThrow('Document type must be one of: txt, md, html');
      });

      it('should throw error for empty content', async () => {
        await expect(repository.create({
          ...mockDocument,
          content: ''
        })).rejects.toThrow('Document content is required');
      });

      it('should throw error for invalid size', async () => {
        await expect(repository.create({
          ...mockDocument,
          size: 0
        })).rejects.toThrow('Document size must be greater than 0');
      });
    });

    describe('findById', () => {
      it('should get document by valid id', async () => {
        const created = await prisma.document.create({
          data: mockDocument
        });

        const document = await repository.findById(created.id);

        expect(document).toBeDefined();
        expect(document?.id).toBe(created.id);
        expect(document?.name).toBe(mockDocument.name);
        expect(document?.type).toBe(mockDocument.type);
      });

      it('should throw error for empty document ID', async () => {
        await expect(repository.findById(''))
          .rejects.toThrow('Document ID is required');
      });

      it('should return null when document not found', async () => {
        const document = await repository.findById('non-existent-id');
        expect(document).toBeNull();
      });
    });

    describe('findByType', () => {
      it('should get documents by valid type', async () => {
        const doc1 = await prisma.document.create({
          data: { ...mockDocument, name: 'doc1.txt' }
        });

        const doc2 = await prisma.document.create({
          data: { ...mockDocument, name: 'doc2.txt' }
        });

        const documents = await repository.findByType('txt');

        expect(documents).toHaveLength(2);
        expect(documents.map(d => d.id)).toContain(doc1.id);
        expect(documents.map(d => d.id)).toContain(doc2.id);
        // Check sorting by createdAt desc
        expect(documents[0].createdAt.getTime()).toBeGreaterThanOrEqual(documents[1].createdAt.getTime());
      });

      it('should throw error for empty type', async () => {
        await expect(repository.findByType(''))
          .rejects.toThrow('Document type is required');
      });

      it('should return empty array when no documents found', async () => {
        const documents = await repository.findByType('md');
        expect(documents).toHaveLength(0);
      });
    });

    describe('findAll', () => {
      it('should get all documents', async () => {
        const doc1 = await prisma.document.create({
          data: { ...mockDocument, name: 'doc1.txt' }
        });

        const doc2 = await prisma.document.create({
          data: { ...mockDocument, name: 'doc2.txt' }
        });

        const documents = await repository.findAll();

        expect(documents).toHaveLength(2);
        expect(documents.map(d => d.id)).toContain(doc1.id);
        expect(documents.map(d => d.id)).toContain(doc2.id);
        // Check sorting by createdAt desc
        expect(documents[0].createdAt.getTime()).toBeGreaterThanOrEqual(documents[1].createdAt.getTime());
      });

      it('should return empty array when no documents exist', async () => {
        const documents = await repository.findAll();
        expect(documents).toHaveLength(0);
      });
    });

    describe('update', () => {
      it('should update document with valid params', async () => {
        const document = await prisma.document.create({
          data: mockDocument
        });

        // Add a small delay to ensure different timestamps
        await new Promise(resolve => setTimeout(resolve, 1000));

        const updateData = {
          name: 'updated.txt',
          content: 'Updated content'
        };

        const updated = await repository.update(document.id, updateData);

        expect(updated).toBeDefined();
        expect(updated.id).toBe(document.id);
        expect(updated.name).toBe(updateData.name);
        expect(updated.content).toBe(updateData.content);
        expect(updated.type).toBe(document.type); // Unchanged field
        expect(updated.updatedAt.getTime()).toBeGreaterThan(document.updatedAt.getTime());
      });

      it('should throw error for empty document ID', async () => {
        await expect(repository.update('', {
          name: 'updated.txt'
        })).rejects.toThrow('Document ID is required');
      });

      it('should throw error for empty name in update', async () => {
        const document = await prisma.document.create({
          data: mockDocument
        });

        await expect(repository.update(document.id, {
          name: ''
        })).rejects.toThrow('Document name cannot be empty');
      });

      it('should throw error for invalid type in update', async () => {
        const document = await prisma.document.create({
          data: mockDocument
        });

        await expect(repository.update(document.id, {
          type: 'pdf'
        })).rejects.toThrow('Document type must be one of: txt, md, html');
      });
    });

    describe('delete', () => {
      it('should delete document', async () => {
        const document = await prisma.document.create({
          data: mockDocument
        });

        const deleted = await repository.delete(document.id);
        expect(deleted.id).toBe(document.id);

        const found = await prisma.document.findUnique({
          where: { id: document.id }
        });
        expect(found).toBeNull();
      });

      it('should throw error for empty document ID', async () => {
        await expect(repository.delete(''))
          .rejects.toThrow('Document ID is required');
      });
    });

    describe('deleteAll', () => {
      it('should delete all documents', async () => {
        await prisma.document.create({
          data: { ...mockDocument, name: 'doc1.txt' }
        });

        await prisma.document.create({
          data: { ...mockDocument, name: 'doc2.txt' }
        });

        await repository.deleteAll();

        const documents = await prisma.document.findMany();
        expect(documents).toHaveLength(0);
      });
    });
  });
});
