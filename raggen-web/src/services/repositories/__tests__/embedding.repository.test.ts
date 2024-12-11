import { PrismaClient, Embedding, Context } from '@prisma/client';
import { EmbeddingRepository } from '../embedding.repository';
import { BaseRepository } from '../base.repository';
import '@testing-library/jest-dom';

describe('EmbeddingRepository', () => {
  let repository: EmbeddingRepository;
  let baseRepository: BaseRepository;

  beforeEach(() => {
    baseRepository = new BaseRepository();
    repository = baseRepository.createRepository(EmbeddingRepository);
  });

  describe('Embedding operations', () => {
    const mockEmbedding = {
      messageId: 1,
      vector: Buffer.from([1, 2, 3, 4]),
      vectorId: 1
    };

    describe('createEmbedding', () => {
      it('should create embedding with valid params', async () => {
        const embedding = await repository.createEmbedding(mockEmbedding);

        expect(embedding).toBeDefined();
        expect(embedding.messageId).toBe(mockEmbedding.messageId);
        expect(embedding.vector).toEqual(mockEmbedding.vector);
        expect(embedding.vectorId).toBe(mockEmbedding.vectorId);
        expect(embedding.id).toBeDefined();
        expect(embedding.createdAt).toBeDefined();
      });

      it('should throw error for invalid message ID', async () => {
        await expect(repository.createEmbedding({
          ...mockEmbedding,
          messageId: 0
        })).rejects.toThrow('Invalid message ID');
      });

      it('should throw error for empty vector', async () => {
        await expect(repository.createEmbedding({
          ...mockEmbedding,
          vector: Buffer.from([])
        })).rejects.toThrow('Vector is required');
      });

      it('should throw error for invalid vector ID', async () => {
        await expect(repository.createEmbedding({
          ...mockEmbedding,
          vectorId: -1
        })).rejects.toThrow('Invalid vector ID');
      });
    });

    describe('getEmbeddingByMessageId', () => {
      it('should get embedding by valid message id', async () => {
        const created = await repository.createEmbedding(mockEmbedding);
        const found = await repository.getEmbeddingByMessageId(mockEmbedding.messageId);

        expect(found).toBeDefined();
        expect(found?.id).toBe(created.id);
        expect(found?.messageId).toBe(mockEmbedding.messageId);
        expect(found?.vector).toEqual(mockEmbedding.vector);
      });

      it('should throw error for invalid message ID', async () => {
        await expect(repository.getEmbeddingByMessageId(0))
          .rejects.toThrow('Invalid message ID');
      });

      it('should return null when embedding not found', async () => {
        const found = await repository.getEmbeddingByMessageId(999);
        expect(found).toBeNull();
      });
    });

    describe('getEmbeddingsByVectorIds', () => {
      it('should get embeddings by valid vector ids', async () => {
        const embedding1 = await repository.createEmbedding({
          ...mockEmbedding,
          vectorId: 1
        });
        const embedding2 = await repository.createEmbedding({
          ...mockEmbedding,
          vectorId: 2
        });

        const embeddings = await repository.getEmbeddingsByVectorIds([1, 2]);

        expect(embeddings).toHaveLength(2);
        expect(embeddings.map(e => e.id)).toContain(embedding1.id);
        expect(embeddings.map(e => e.id)).toContain(embedding2.id);
        // Проверяем, что включены связи с сообщениями
        expect(embeddings[0].message).toBeDefined();
        expect(embeddings[1].message).toBeDefined();
      });

      it('should throw error for empty vector IDs array', async () => {
        await expect(repository.getEmbeddingsByVectorIds([]))
          .rejects.toThrow('Vector IDs array is empty');
      });

      it('should throw error for invalid vector IDs', async () => {
        await expect(repository.getEmbeddingsByVectorIds([1, -1]))
          .rejects.toThrow('Invalid vector ID in array');
      });
    });
  });

  describe('Context operations', () => {
    const mockContext = {
      messageId: 1,
      sourceId: 2,
      score: 0.9,
      usedInPrompt: true
    };

    describe('createContext', () => {
      it('should create context with valid params', async () => {
        const context = await repository.createContext(mockContext);

        expect(context).toBeDefined();
        expect(context.messageId).toBe(mockContext.messageId);
        expect(context.sourceId).toBe(mockContext.sourceId);
        expect(context.score).toBe(mockContext.score);
        expect(context.usedInPrompt).toBe(mockContext.usedInPrompt);
        expect(context.id).toBeDefined();
        expect(context.createdAt).toBeDefined();
      });

      it('should throw error for invalid message ID', async () => {
        await expect(repository.createContext({
          ...mockContext,
          messageId: 0
        })).rejects.toThrow('Invalid message ID');
      });

      it('should throw error for invalid source ID', async () => {
        await expect(repository.createContext({
          ...mockContext,
          sourceId: -1
        })).rejects.toThrow('Invalid source ID');
      });

      it('should throw error for invalid score', async () => {
        await expect(repository.createContext({
          ...mockContext,
          score: 1.5
        })).rejects.toThrow('Score must be between 0 and 1');
      });

      it('should throw error for invalid usedInPrompt', async () => {
        await expect(repository.createContext({
          ...mockContext,
          usedInPrompt: 'true' as any
        })).rejects.toThrow('usedInPrompt must be a boolean');
      });
    });

    describe('createManyContexts', () => {
      it('should create multiple contexts with valid params', async () => {
        const contexts = [
          mockContext,
          { ...mockContext, sourceId: 3 }
        ];

        await repository.createManyContexts(contexts);

        const found = await repository.getContextByMessageId(mockContext.messageId);
        expect(found).toHaveLength(2);
        expect(found.map(c => c.sourceId)).toContain(2);
        expect(found.map(c => c.sourceId)).toContain(3);
      });

      it('should throw error for empty contexts array', async () => {
        await expect(repository.createManyContexts([]))
          .rejects.toThrow('Contexts array is empty');
      });

      it('should throw error if any context is invalid', async () => {
        await expect(repository.createManyContexts([
          mockContext,
          { ...mockContext, score: 1.5 }
        ])).rejects.toThrow('Score must be between 0 and 1');
      });
    });

    describe('getContextByMessageId', () => {
      it('should get contexts by valid message id', async () => {
        const created = await repository.createContext(mockContext);
        const found = await repository.getContextByMessageId(mockContext.messageId);

        expect(found).toHaveLength(1);
        expect(found[0].id).toBe(created.id);
        expect(found[0].messageId).toBe(mockContext.messageId);
        expect(found[0].sourceId).toBe(mockContext.sourceId);
        expect(found[0].score).toBe(mockContext.score);
        // Проверяем, что включены связи с сообщениями
        expect(found[0].message).toBeDefined();
      });

      it('should throw error for invalid message ID', async () => {
        await expect(repository.getContextByMessageId(0))
          .rejects.toThrow('Invalid message ID');
      });

      it('should return empty array when no contexts found', async () => {
        const found = await repository.getContextByMessageId(999);
        expect(found).toHaveLength(0);
      });

      it('should sort contexts by score in descending order', async () => {
        await repository.createContext({ ...mockContext, score: 0.8 });
        await repository.createContext({ ...mockContext, score: 0.9 });
        await repository.createContext({ ...mockContext, score: 0.7 });

        const contexts = await repository.getContextByMessageId(mockContext.messageId);

        expect(contexts).toHaveLength(3);
        expect(contexts[0].score).toBe(0.9);
        expect(contexts[1].score).toBe(0.8);
        expect(contexts[2].score).toBe(0.7);
      });
    });
  });
});
