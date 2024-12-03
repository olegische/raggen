import { ProviderFactory, ProviderType } from '@/providers/factory';

export class ModelService {
  private static modelCache: Map<ProviderType, string[]> = new Map();
  private static cacheTimeout = 5 * 60 * 1000; // 5 минут
  private static lastUpdate: Map<ProviderType, number> = new Map();

  static async getModels(providerType: ProviderType): Promise<string[]> {
    const now = Date.now();
    const lastUpdate = this.lastUpdate.get(providerType) || 0;

    // Проверяем кэш
    if (
      this.modelCache.has(providerType) && 
      now - lastUpdate < this.cacheTimeout
    ) {
      return this.modelCache.get(providerType)!;
    }

    try {
      const provider = ProviderFactory.createProvider(providerType);
      const models = await provider.listModels();

      // Обновляем кэш
      this.modelCache.set(providerType, models);
      this.lastUpdate.set(providerType, now);

      return models;
    } catch (error) {
      console.error(`Error fetching models for ${providerType}:`, error);
      throw error;
    }
  }

  static clearCache(providerType?: ProviderType) {
    if (providerType) {
      this.modelCache.delete(providerType);
      this.lastUpdate.delete(providerType);
    } else {
      this.modelCache.clear();
      this.lastUpdate.clear();
    }
  }
} 