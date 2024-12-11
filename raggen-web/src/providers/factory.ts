import { BaseProvider } from './base.provider';
import { YandexGPTProvider } from './yandex/provider';
import { GigaChatProvider } from './gigachat/provider';
import { ProviderType, getProviderConfig, PROVIDER_CONFIG } from '../config/providers';

export type { ProviderType };

export interface ProviderInfo {
  id: ProviderType;
  displayName: string;
  systemPrompt: string;
}

export class ProviderFactory {
  static createProvider(type: ProviderType): BaseProvider {
    console.log(`Creating provider for type: ${type}`);
    const config = getProviderConfig(type);
    
    switch (type) {
      case 'yandex':
        console.log('Creating YandexGPT provider with URL:', config.apiUrl);
        return new YandexGPTProvider({
          apiUrl: config.apiUrl,
          credentials: config.credentials,
          systemPrompt: config.systemPrompt
        });
      case 'gigachat':
        console.log('Creating GigaChat provider with URL:', config.apiUrl);
        return new GigaChatProvider({
          apiUrl: config.apiUrl,
          credentials: config.credentials,
          systemPrompt: config.systemPrompt
        });
      default:
        throw new Error(`Unknown provider: ${type}`);
    }
  }

  static getProvider(type: ProviderType): BaseProvider {
    return this.createProvider(type);
  }

  static validateProvider(type: string): boolean {
    return type in PROVIDER_CONFIG;
  }

  static getSupportedProviders(): ProviderInfo[] {
    return Object.values(PROVIDER_CONFIG);
  }
}
