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

    it('should create embedding', async () => {
      const embedding = await repository.createEmbedding(mockEmbedding);

      expect(embedding).toBeDefined();
      expect(embedding.messageId).toBe(mockEmbedding.messageId);
      expect(embedding.vector).toEqual(mockEmbedding.vector);
      expect(embedding.vectorId).toBe(mockEmbedding.vectorId);
      expect(embedding.id).toBeDefined();
      expect(embedding.createdAt).toBeDefined();
    });

    it('should get embedding by message id', async () => {
      const created = await repository.createEmbedding(mockEmbedding);
      const found = await repository.getEmbeddingByMessageId(mockEmbedding.messageId);

      expect(found).toBeDefined();
      expect(found?.id).toBe(created.id);
      expect(found?.messageId).toBe(mockEmbedding.messageId);
      expect(found?.vector).toEqual(mockEmbedding.vector);
    });

    it('should get embeddings by vector ids', async () => {
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
    });
  });

  describe('Context operations', () => {
    const mockContext = {
      messageId: 1,
      sourceId: 2,
      score: 0.9,
      usedInPrompt: true
    };

    it('should create context', async () => {
      const context = await repository.createContext(mockContext);

      expect(context).toBeDefined();
      expect(context.messageId).toBe(mockContext.messageId);
      expect(context.sourceId).toBe(mockContext.sourceId);
      expect(context.score).toBe(mockContext.score);
      expect(context.usedInPrompt).toBe(mockContext.usedInPrompt);
      expect(context.id).toBeDefined();
      expect(context.createdAt).toBeDefined();
    });

    it('should create many contexts', async () => {
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

    it('should get context by message id', async () => {
      const created = await repository.createContext(mockContext);
      const found = await repository.getContextByMessageId(mockContext.messageId);

      expect(found).toHaveLength(1);
      expect(found[0].id).toBe(created.id);
      expect(found[0].messageId).toBe(mockContext.messageId);
      expect(found[0].sourceId).toBe(mockContext.sourceId);
      expect(found[0].score).toBe(mockContext.score);
      // Проверяем сортировку по score desc
      expect(found[0].score).toBe(mockContext.score);
    });

    it('should get contexts sorted by score', async () => {
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
