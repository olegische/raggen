import { ProviderFactory, ProviderType } from '../factory';
import { YandexGPTProvider } from '../yandex/provider';
import { GigaChatProvider } from '../gigachat/provider';

// Мокаем переменные окружения
process.env.YANDEX_GPT_API_URL = 'https://yandex.api';
process.env.YANDEX_API_KEY = 'yandex-key';
process.env.GIGACHAT_API_URL = 'https://gigachat.api';
process.env.GIGACHAT_CREDENTIALS = 'gigachat-credentials';

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
    }).toThrow('Unknown provider type: unknown');
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
      credentials: 'yandex-key'
    });

    // @ts-expect-error: Accessing protected property for testing
    expect(gigachatProvider.config).toEqual({
      apiUrl: 'https://gigachat.api',
      credentials: 'gigachat-credentials'
    });
  });
}); 