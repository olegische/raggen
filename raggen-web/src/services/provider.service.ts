import { ProviderFactory, ProviderType } from '@/providers/factory';

interface ProviderStatus {
  available: boolean;
  lastCheck: number;
  error?: string;
}

export class ProviderService {
  private static statusCache: Map<ProviderType, ProviderStatus> = new Map();
  private static checkInterval = 60 * 1000; // 1 минута

  static async getAvailableProviders(): Promise<ProviderType[]> {
    const providers: ProviderType[] = ['yandex', 'gigachat'];
    const available: ProviderType[] = [];

    for (const provider of providers) {
      console.log(`Checking availability of provider: ${provider}`);
      const isAvailable = await this.isProviderAvailable(provider);
      console.log(`Provider ${provider} availability:`, isAvailable);
      if (isAvailable) {
        available.push(provider);
      }
    }

    console.log('Available providers:', available);
    return available;
  }

  static async isProviderAvailable(type: ProviderType): Promise<boolean> {
    const now = Date.now();
    const status = this.statusCache.get(type);

    // Проверяем кэш
    if (status && now - status.lastCheck < this.checkInterval) {
      console.log(`Using cached status for ${type}:`, status);
      return status.available;
    }

    try {
      console.log(`Creating provider instance for ${type}`);
      const provider = ProviderFactory.createProvider(type);
      console.log(`Listing models for ${type}`);
      await provider.listModels();

      const newStatus = {
        available: true,
        lastCheck: now
      };
      console.log(`Setting status for ${type}:`, newStatus);
      this.statusCache.set(type, newStatus);

      return true;
    } catch (error) {
      console.error(`Error checking ${type} availability:`, error);
      const newStatus = {
        available: false,
        lastCheck: now,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
      console.log(`Setting error status for ${type}:`, newStatus);
      this.statusCache.set(type, newStatus);

      return false;
    }
  }

  static getProviderStatus(type: ProviderType): ProviderStatus | undefined {
    return this.statusCache.get(type);
  }

  static clearCache(type?: ProviderType) {
    if (type) {
      this.statusCache.delete(type);
    } else {
      this.statusCache.clear();
    }
  }
} 