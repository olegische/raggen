import { ProviderFactory } from '../factory';
import { YandexGPTProvider } from '../yandex/provider';
import { GigaChatProvider } from '../gigachat/provider';
import { ProviderType } from '../../config/providers';

// Мокаем конфигурацию провайдеров
jest.mock('../../config/providers', () => ({
  ProviderType: {
    yandex: 'yandex',
    gigachat: 'gigachat'
  },
  PROVIDER_CONFIG: {
    yandex: {
      id: 'yandex',
      displayName: 'YandexGPT',
      systemPrompt: 'yandex',
      apiUrl: 'https://yandex.api',
      credentials: 'yandex-key'
    },
    gigachat: {
      id: 'gigachat',
      displayName: 'GigaChat',
      systemPrompt: 'gigachat',
      apiUrl: 'https://gigachat.api',
      credentials: 'gigachat-credentials'
    }
  },
  getProviderConfig: (type: string) => {
    const config = {
      yandex: {
        id: 'yandex',
        displayName: 'YandexGPT',
        systemPrompt: 'yandex',
        apiUrl: 'https://yandex.api',
        credentials: 'yandex-key'
      },
      gigachat: {
        id: 'gigachat',
        displayName: 'GigaChat',
        systemPrompt: 'gigachat',
        apiUrl: 'https://gigachat.api',
        credentials: 'gigachat-credentials'
      }
    }[type];
    
    if (!config) {
      throw new Error(`Unknown provider: ${type}`);
    }
    return config;
  }
}));

describe('ProviderFactory', () => {
  const providers: ProviderType[] = ['yandex', 'gigachat'];

  test.each(providers)('should create %s provider', (type) => {
    const provider = ProviderFactory.createProvider(type);
    
    switch (type) {
      case 'yandex':
        expect(provider).toBeInstanceOf(YandexGPTProvider);
        break;
      case 'gigachat':
        expect(provider).toBeInstanceOf(GigaChatProvider);
        break;
    }
  });

  test('should throw error for unknown provider', () => {
    expect(() => {
      // @ts-expect-error: Testing invalid provider type
      ProviderFactory.createProvider('unknown');
    }).toThrow('Unknown provider: unknown');
  });

  test('providers should have required methods', () => {
    providers.forEach(type => {
      const provider = ProviderFactory.createProvider(type);
      expect(provider.generateResponse).toBeDefined();
      expect(provider.listModels).toBeDefined();
      expect(typeof provider.generateResponse).toBe('function');
      expect(typeof provider.listModels).toBe('function');
    });
  });

  test('providers should be properly configured', () => {
    const yandexProvider = ProviderFactory.createProvider('yandex');
    const gigachatProvider = ProviderFactory.createProvider('gigachat');

    // @ts-expect-error: Accessing protected property for testing
    expect(yandexProvider.config).toEqual({
      apiUrl: 'https://yandex.api',
      credentials: 'yandex-key',
      systemPrompt: 'yandex'
    });

    // @ts-expect-error: Accessing protected property for testing
    expect(gigachatProvider.config).toEqual({
      apiUrl: 'https://gigachat.api',
      credentials: 'gigachat-credentials',
      systemPrompt: 'gigachat'
    });
  });
});
