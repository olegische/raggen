import { ProviderService } from '../provider.service';
import { ProviderFactory } from '@/providers/factory';
import { BaseProvider } from '@/providers/base.provider';
import { z } from 'zod';

// Мокаем фабрику провайдеров
jest.mock('@/providers/factory');
const mockedFactory = jest.mocked(ProviderFactory);

describe('ProviderService', () => {
  // Создаем мок провайдера
  const mockProvider = {
    constructor: { name: 'YandexProvider' },
    generateResponse: jest.fn(),
    listModels: jest.fn(),
    config: { apiUrl: 'test', credentials: 'test' },
    responseSchema: z.any(),
    validateResponse: jest.fn(),
    formatError: jest.fn(),
    validateOptions: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
    mockedFactory.createProvider.mockReturnValue(mockProvider as unknown as BaseProvider);
    ProviderService.clearCache();
  });

  describe('getAvailableProviders', () => {
    it('should return all available providers', async () => {
      mockProvider.listModels.mockResolvedValue(['model1']);

      const providers = await ProviderService.getAvailableProviders();

      expect(providers).toContain('yandex');
      expect(providers).toContain('gigachat');
      expect(mockedFactory.createProvider).toHaveBeenCalledTimes(2);
    });

    it('should filter out unavailable providers', async () => {
      mockProvider.listModels
        .mockResolvedValueOnce(['model1']) // yandex доступен
        .mockRejectedValueOnce(new Error('API Error')); // gigachat недоступен

      const providers = await ProviderService.getAvailableProviders();

      expect(providers).toContain('yandex');
      expect(providers).not.toContain('gigachat');
    });
  });

  describe('isProviderAvailable', () => {
    it('should return true for working provider', async () => {
      mockProvider.listModels.mockResolvedValue(['model1']);

      const isAvailable = await ProviderService.isProviderAvailable('yandex');

      expect(isAvailable).toBe(true);
      expect(mockedFactory.createProvider).toHaveBeenCalledWith('yandex');
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
      const checkInterval = ProviderService['checkInterval'];
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