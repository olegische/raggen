import { ModelService } from '../model.service';
import { ProviderFactory } from '@/providers/factory';
import { BaseProvider } from '@/providers/base.provider';
import { z } from 'zod';

// Мокаем фабрику провайдеров
jest.mock('@/providers/factory');
const mockedFactory = jest.mocked(ProviderFactory);

describe('ModelService', () => {
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
    ModelService.clearCache(); // Очищаем кэш перед каждым тестом
  });

  describe('getModels', () => {
    const mockModels = ['model1', 'model2', 'model3'];

    beforeEach(() => {
      mockProvider.listModels.mockResolvedValue(mockModels);
    });

    it('should fetch models from provider', async () => {
      const result = await ModelService.getModels('yandex');

      expect(mockedFactory.createProvider).toHaveBeenCalledWith('yandex');
      expect(mockProvider.listModels).toHaveBeenCalled();
      expect(result).toEqual(mockModels);
    });

    it('should cache models', async () => {
      // Первый запрос
      await ModelService.getModels('yandex');
      
      // Второй запрос должен использовать кэш
      await ModelService.getModels('yandex');

      expect(mockProvider.listModels).toHaveBeenCalledTimes(1);
    });

    it('should refresh cache after timeout', async () => {
      // Устанавливаем маленький timeout для теста
      jest.useFakeTimers();
      const originalTimeout = ModelService['cacheTimeout'];
      ModelService['cacheTimeout'] = 1000; // 1 секунда

      // Первый запрос
      await ModelService.getModels('yandex');
      
      // Продвигаем время вперед
      jest.advanceTimersByTime(1500);

      // Второй запрос должен обновить кэш
      await ModelService.getModels('yandex');

      expect(mockProvider.listModels).toHaveBeenCalledTimes(2);

      // Восстанавливаем оригинальный timeout
      ModelService['cacheTimeout'] = originalTimeout;
      jest.useRealTimers();
    });

    it('should handle different providers separately', async () => {
      // Запрос для первого провайдера
      await ModelService.getModels('yandex');
      
      // Запрос для второго провайдера
      await ModelService.getModels('gigachat');

      expect(mockedFactory.createProvider).toHaveBeenCalledWith('yandex');
      expect(mockedFactory.createProvider).toHaveBeenCalledWith('gigachat');
      expect(mockProvider.listModels).toHaveBeenCalledTimes(2);
    });

    it('should handle provider errors', async () => {
      mockProvider.listModels.mockRejectedValue(new Error('Provider error'));

      await expect(ModelService.getModels('yandex'))
        .rejects
        .toThrow('Provider error');
    });
  });

  describe('clearCache', () => {
    beforeEach(() => {
      mockProvider.listModels.mockResolvedValue(['model1']);
    });

    it('should clear cache for specific provider', async () => {
      // Заполняем кэш
      await ModelService.getModels('yandex');
      await ModelService.getModels('gigachat');

      // Очищаем кэш только для yandex
      ModelService.clearCache('yandex');

      // Запрос для yandex должен обновить кэш
      await ModelService.getModels('yandex');
      // Запрос для gigachat должен использовать кэш
      await ModelService.getModels('gigachat');

      expect(mockProvider.listModels).toHaveBeenCalledTimes(3);
    });

    it('should clear entire cache', async () => {
      // Заполняем кэш
      await ModelService.getModels('yandex');
      await ModelService.getModels('gigachat');

      // Очищаем весь кэш
      ModelService.clearCache();

      // Оба запроса должны обновить кэш
      await ModelService.getModels('yandex');
      await ModelService.getModels('gigachat');

      expect(mockProvider.listModels).toHaveBeenCalledTimes(4);
    });
  });
}); 