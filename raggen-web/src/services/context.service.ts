import { DatabaseService } from './database';
import { EmbedApiClient } from './embed-api';
import { Message, Context } from '@prisma/client';

export interface ContextSearchResult {
  message: Message;
  score: number;
  usedInPrompt: boolean;
}

export interface ContextSearchOptions {
  maxResults?: number;
  minScore?: number;
  excludeMessageIds?: number[];
}

export class ContextService {
  private static readonly DEFAULT_MAX_RESULTS = 5;
  private static readonly DEFAULT_MIN_SCORE = 0.7;
  private static readonly CACHE_TTL = 5 * 60 * 1000; // 5 minutes

  private contextCache: Map<string, { results: ContextSearchResult[]; timestamp: number }>;

  constructor(
    private readonly database: DatabaseService,
    public readonly embedApi: EmbedApiClient
  ) {
    this.contextCache = new Map();
  }

  /**
   * Поиск релевантного контекста для сообщения
   */
  async searchContext(
    text: string,
    options: ContextSearchOptions = {}
  ): Promise<ContextSearchResult[]> {
    const cacheKey = this.getCacheKey(text, options);
    const cachedResults = this.getFromCache(cacheKey);
    if (cachedResults) {
      return cachedResults;
    }

    const {
      maxResults = ContextService.DEFAULT_MAX_RESULTS,
      minScore = ContextService.DEFAULT_MIN_SCORE,
      excludeMessageIds = []
    } = options;

    // Поиск похожих текстов через API эмбеддингов
    const searchResponse = await this.embedApi.searchSimilar(text, maxResults * 2);

    // Фильтруем результаты по score и исключаем указанные сообщения
    const filteredResults = searchResponse.results.filter(
      result => result.score >= minScore && !excludeMessageIds.includes(result.vector_id)
    );

    // Получаем сообщения по vector_id
    const embeddings = await this.database.getEmbeddingsByVectorIds(
      filteredResults.slice(0, maxResults).map(r => r.vector_id)
    );

    // Получаем сообщения из базы данных
    const messages = await this.database.getMessagesByIds(
      embeddings.map(e => e.messageId)
    );

    // Формируем результаты с сообщениями и оценками
    const results = embeddings.map(embedding => {
      const searchResult = filteredResults.find(r => r.vector_id === embedding.vectorId);
      const message = messages.find(m => m.id === embedding.messageId);
      if (!message) {
        throw new Error(`Message not found for embedding ${embedding.id}`);
      }
      return {
        message,
        score: searchResult?.score || 0,
        usedInPrompt: false
      };
    });

    // Сортируем по релевантности
    const sortedResults = this.prioritizeResults(results);

    // Кэшируем результаты
    this.cacheResults(cacheKey, sortedResults);

    return sortedResults;
  }

  /**
   * Сохранение использованного контекста
   */
  async saveUsedContext(
    messageId: number,
    contexts: ContextSearchResult[]
  ): Promise<void> {
    const contextData = contexts.map(context => ({
      messageId,
      sourceId: context.message.id,
      score: context.score,
      usedInPrompt: context.usedInPrompt
    }));

    await this.database.createManyContexts(contextData);
  }

  /**
   * Приоритизация результатов поиска
   */
  private prioritizeResults(results: ContextSearchResult[]): ContextSearchResult[] {
    return results.sort((a, b) => {
      // Основной критерий - оценка релевантности
      const scoreDiff = b.score - a.score;
      if (Math.abs(scoreDiff) > 0.1) {
        return scoreDiff;
      }

      // При близких оценках учитываем время создания сообщения
      return b.message.timestamp.getTime() - a.message.timestamp.getTime();
    });
  }

  /**
   * Получение ключа кэша
   */
  private getCacheKey(text: string, options: ContextSearchOptions): string {
    return `${text}:${JSON.stringify(options)}`;
  }

  /**
   * Получение результатов из кэша
   */
  private getFromCache(key: string): ContextSearchResult[] | null {
    const cached = this.contextCache.get(key);
    if (!cached) {
      return null;
    }

    // Проверяем TTL
    if (Date.now() - cached.timestamp > ContextService.CACHE_TTL) {
      this.contextCache.delete(key);
      return null;
    }

    return cached.results;
  }

  /**
   * Сохранение результатов в кэш
   */
  private cacheResults(key: string, results: ContextSearchResult[]): void {
    this.contextCache.set(key, {
      results,
      timestamp: Date.now()
    });
  }
} 