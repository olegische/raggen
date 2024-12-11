import { ProviderService } from '../provider.service';
import { ProviderFactory } from '../../providers/factory';
import { BaseProvider } from '../../providers/base.provider';
import { z } from 'zod';

// Мокаем фабрику провайдеров
jest.mock('../../providers/factory');
const mockedFactory = jest.mocked(ProviderFactory);

describe('ProviderService', () => {
  // Создаем мок провайдера
  const mockProvider = {
    constructor: { name: 'YandexProvider' },
    generateResponse: jest.fn(),
    listModels: jest.fn(),
    config: { apiUrl: 'test', credentials: 'test', systemPrompt: 'yandex' },
    responseSchema: z.any(),
    validateResponse: jest.fn(),
    formatError: jest.fn(),
    validateOptions: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockedFactory.getProvider.mockReturnValue(mockProvider as unknown as BaseProvider);
    mockedFactory.getSupportedProviders.mockReturnValue([
      { id: 'yandex', displayName: 'YandexGPT', systemPrompt: 'yandex' },
      { id: 'gigachat', displayName: 'GigaChat', systemPrompt: 'gigachat' }
    ]);
    ProviderService.clearCache();
  });

  describe('getAvailableProviders', () => {
    it('should return all available providers', async () => {
      mockProvider.listModels.mockResolvedValue(['model1']);

      const providers = await ProviderService.getAvailableProviders();

      expect(providers).toHaveLength(2);
      expect(providers[0].id).toBe('yandex');
      expect(providers[1].id).toBe('gigachat');
      expect(providers[0].status.available).toBe(true);
      expect(providers[1].status.available).toBe(true);
      expect(mockedFactory.getSupportedProviders).toHaveBeenCalled();
    });

    it('should filter out unavailable providers', async () => {
      mockProvider.listModels
        .mockResolvedValueOnce(['model1']) // yandex доступен
        .mockRejectedValueOnce(new Error('API Error')); // gigachat недоступен

      const providers = await ProviderService.getAvailableProviders();

      expect(providers).toHaveLength(2);
      expect(providers[0].status.available).toBe(true);
      expect(providers[1].status.available).toBe(false);
    });
  });

  describe('isProviderAvailable', () => {
    it('should return true for working provider', async () => {
      mockProvider.listModels.mockResolvedValue(['model1']);

      const isAvailable = await ProviderService.isProviderAvailable('yandex');

      expect(isAvailable).toBe(true);
      expect(mockedFactory.getProvider).toHaveBeenCalledWith('yandex');
    });

    it('should return false for failing provider', async () => {
      mockProvider.listModels.mockRejectedValue(new Error('API Error'));

      const isAvailable = await ProviderService.isProviderAvailable('yandex');

      expect(isAvailable).toBe(false);
    });

    it('should cache provider status', async () => {
      mockProvider.listModels.mockResolvedValue(['model1']);

      // Первая проверка
      await ProviderService.isProviderAvailable('yandex');
      // Вторая проверка должна использовать кэш
      await ProviderService.isProviderAvailable('yandex');

      expect(mockProvider.listModels).toHaveBeenCalledTimes(1);
    });

    it('should refresh status after interval', async () => {
      jest.useFakeTimers();
      const checkInterval = 60 * 1000; // 1 минута
      mockProvider.listModels.mockResolvedValue(['model1']);

      // Первая проверка
      await ProviderService.isProviderAvailable('yandex');
      
      // Продвигаем время вперед
      jest.advanceTimersByTime(checkInterval + 100);

      // Вторая проверка должна обновить статус
      await ProviderService.isProviderAvailable('yandex');

      expect(mockProvider.listModels).toHaveBeenCalledTimes(2);
      jest.useRealTimers();
    });
  });

  describe('getProviderStatus', () => {
    it('should return cached status', async () => {
      mockProvider.listModels.mockResolvedValue(['model1']);

      await ProviderService.isProviderAvailable('yandex');
      const status = ProviderService.getProviderStatus('yandex');

      expect(status).toBeDefined();
      expect(status?.available).toBe(true);
      expect(status?.lastCheck).toBeDefined();
    });

    it('should return undefined for unchecked provider', () => {
      const status = ProviderService.getProviderStatus('yandex');

      expect(status).toBeUndefined();
    });

    it('should include error message for failing provider', async () => {
      const errorMessage = 'API Error';
      mockProvider.listModels.mockRejectedValue(new Error(errorMessage));

      await ProviderService.isProviderAvailable('yandex');
      const status = ProviderService.getProviderStatus('yandex');

      expect(status?.available).toBe(false);
      expect(status?.error).toBe(errorMessage);
    });
  });

  describe('clearCache', () => {
    beforeEach(async () => {
      mockProvider.listModels.mockResolvedValue(['model1']);
      await ProviderService.isProviderAvailable('yandex');
      await ProviderService.isProviderAvailable('gigachat');
    });

    it('should clear status for specific provider', async () => {
      ProviderService.clearCache('yandex');

      expect(ProviderService.getProviderStatus('yandex')).toBeUndefined();
      expect(ProviderService.getProviderStatus('gigachat')).toBeDefined();
    });

    it('should clear all statuses', () => {
      ProviderService.clearCache();

      expect(ProviderService.getProviderStatus('yandex')).toBeUndefined();
      expect(ProviderService.getProviderStatus('gigachat')).toBeUndefined();
    });
  });
});
