import { PrismaClient } from '@prisma/client';
import { mockDeep, mockReset, DeepMockProxy } from 'jest-mock-extended';
import { EmbeddingRepository } from '../embedding.repository';

describe('EmbeddingRepository', () => {
  let prisma: DeepMockProxy<PrismaClient>;
  let repository: EmbeddingRepository;
  const mockDate = new Date('2024-01-01T10:00:00.000Z');

  const mockMessage = {
    id: 1,
    chatId: 'chat-1',
    message: 'Test message',
    response: null,
    model: 'test-model',
    provider: 'test-provider',
    timestamp: mockDate,
    temperature: 0.7,
    maxTokens: 100
  };

  const mockDocument = {
    id: 'doc-1',
    name: 'test.txt',
    type: 'txt',
    size: 1024,
    content: 'Test content',
    metadata: null,
    createdAt: mockDate,
    updatedAt: mockDate
  };

  const mockEmbedding = {
    id: 1,
    messageId: 1,
    documentId: null,
    vector: Buffer.from('test'),
    vectorId: 1,
    createdAt: mockDate
  };

  const mockContext = {
    id: 1,
    messageId: 1,
    sourceId: 2,
    documentId: null,
    score: 0.8,
    usedInPrompt: true,
    createdAt: mockDate
  };

  beforeEach(() => {
    prisma = mockDeep<PrismaClient>();
    repository = new EmbeddingRepository(prisma);
    mockReset(prisma);
  });

  describe('createEmbedding', () => {
    it('should create embedding for message', async () => {
      prisma.message.findUnique.mockResolvedValue(mockMessage);
      prisma.embedding.create.mockResolvedValue(mockEmbedding);

      const params = {
        messageId: 1,
        vector: Buffer.from('test'),
        vectorId: 1
      };

      const result = await repository.createEmbedding(params);
      expect(result).toEqual(mockEmbedding);
    });

    it('should create embedding for document', async () => {
      const documentEmbedding = { ...mockEmbedding, messageId: null, documentId: 'doc-1' };
      prisma.document.findUnique.mockResolvedValue(mockDocument);
      prisma.embedding.create.mockResolvedValue(documentEmbedding);

      const params = {
        documentId: 'doc-1',
        vector: Buffer.from('test'),
        vectorId: 1
      };

      const result = await repository.createEmbedding(params);
      expect(result).toEqual(documentEmbedding);
    });

    it('should throw error if neither messageId nor documentId provided', async () => {
      const params = {
        vector: Buffer.from('test'),
        vectorId: 1
      };

      await expect(repository.createEmbedding(params)).rejects.toThrow('Either message ID or document ID is required');
    });

    it('should throw error if both messageId and documentId provided', async () => {
      const params = {
        messageId: 1,
        documentId: 'doc-1',
        vector: Buffer.from('test'),
        vectorId: 1
      };

      await expect(repository.createEmbedding(params)).rejects.toThrow('Cannot create embedding for both message and document');
    });
  });

  describe('getEmbeddingByMessageId', () => {
    it('should get embedding by message ID', async () => {
      prisma.message.findUnique.mockResolvedValue(mockMessage);
      prisma.embedding.findUnique.mockResolvedValue(mockEmbedding);

      const result = await repository.getEmbeddingByMessageId(1);
      expect(result).toEqual(mockEmbedding);
    });

    it('should return null if message not found', async () => {
      prisma.message.findUnique.mockResolvedValue(null);

      const result = await repository.getEmbeddingByMessageId(1);
      expect(result).toBeNull();
    });
  });

  describe('getEmbeddingByDocumentId', () => {
    it('should get embedding by document ID', async () => {
      const documentEmbedding = { ...mockEmbedding, messageId: null, documentId: 'doc-1' };
      prisma.document.findUnique.mockResolvedValue(mockDocument);
      prisma.embedding.findUnique.mockResolvedValue(documentEmbedding);

      const result = await repository.getEmbeddingByDocumentId('doc-1');
      expect(result).toEqual(documentEmbedding);
    });

    it('should return null if document not found', async () => {
      prisma.document.findUnique.mockResolvedValue(null);

      const result = await repository.getEmbeddingByDocumentId('doc-1');
      expect(result).toBeNull();
    });
  });

  describe('getEmbeddingsByVectorIds', () => {
    it('should get embeddings by vector IDs including both messages and documents', async () => {
      const messageEmbedding = {
        ...mockEmbedding,
        message: mockMessage,
        document: null
      };
      const documentEmbedding = {
        ...mockEmbedding,
        messageId: null,
        documentId: 'doc-1',
        message: null,
        document: mockDocument
      };

      prisma.embedding.findMany.mockResolvedValue([messageEmbedding, documentEmbedding]);

      const result = await repository.getEmbeddingsByVectorIds([1, 2]);
      expect(result).toHaveLength(2);
      expect(result).toContainEqual(expect.objectContaining({ messageId: 1 }));
      expect(result).toContainEqual(expect.objectContaining({ documentId: 'doc-1' }));
    });

    it('should filter out embeddings without message or document', async () => {
      const invalidEmbedding = {
        ...mockEmbedding,
        message: null,
        document: null
      };

      prisma.embedding.findMany.mockResolvedValue([invalidEmbedding]);

      const result = await repository.getEmbeddingsByVectorIds([1]);
      expect(result).toHaveLength(0);
    });
  });

  describe('createContext', () => {
    it('should create context with document reference', async () => {
      const contextWithDoc = { ...mockContext, documentId: 'doc-1' };
      prisma.message.findUnique.mockResolvedValueOnce(mockMessage);
      prisma.message.findUnique.mockResolvedValueOnce(mockMessage);
      prisma.document.findUnique.mockResolvedValue(mockDocument);
      prisma.context.create.mockResolvedValue(contextWithDoc);

      const params = {
        messageId: 1,
        sourceId: 2,
        documentId: 'doc-1',
        score: 0.8,
        usedInPrompt: true
      };

      const result = await repository.createContext(params);
      expect(result).toEqual(contextWithDoc);
    });
  });

  describe('getContextByDocumentId', () => {
    it('should get contexts by document ID', async () => {
      const contextWithDoc = {
        ...mockContext,
        documentId: 'doc-1',
        message: mockMessage,
        document: mockDocument
      };

      prisma.document.findUnique.mockResolvedValue(mockDocument);
      prisma.context.findMany.mockResolvedValue([contextWithDoc]);

      const result = await repository.getContextByDocumentId('doc-1');
      expect(result).toHaveLength(1);
      expect(result[0]).toEqual(contextWithDoc);
    });

    it('should return empty array if document not found', async () => {
      prisma.document.findUnique.mockResolvedValue(null);

      const result = await repository.getContextByDocumentId('doc-1');
      expect(result).toHaveLength(0);
    });
  });
});
