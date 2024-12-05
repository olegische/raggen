import axios, { AxiosError } from 'axios';
import { EmbedApiClient } from '../embed-api';

// Мокаем axios
jest.mock('axios', () => ({
  create: jest.fn(() => ({
    post: jest.fn(),
    interceptors: {
      response: {
        use: jest.fn(),
      },
    },
  })),
  isAxiosError: jest.fn((error: any): error is AxiosError => true),
}));

describe('EmbedApiClient', () => {
  let client: EmbedApiClient;
  let axiosInstance: any;

  beforeEach(() => {
    // Очищаем моки перед каждым тестом
    jest.clearAllMocks();
    
    // Создаем новый экземпляр клиента
    client = new EmbedApiClient('http://test-api.com', 3, 100);
    
    // Получаем мок axios instance
    axiosInstance = (axios.create as jest.Mock).mock.results[0].value;
  });

  describe('constructor', () => {
    it('should create axios instance with correct config', () => {
      expect(axios.create).toHaveBeenCalledWith({
        baseURL: 'http://test-api.com',
        timeout: 10000,
        headers: {
          'Content-Type': 'application/json',
        },
      });
    });

    it('should use default values if not provided', () => {
      client = new EmbedApiClient();
      expect(axios.create).toHaveBeenCalledWith(expect.objectContaining({
        baseURL: 'http://localhost:8001',
      }));
    });
  });

  describe('embedText', () => {
    const mockResponse = {
      embedding: [0.1, 0.2, 0.3],
      text: 'test text',
      vector_id: 1,
    };

    it('should successfully embed single text', async () => {
      axiosInstance.post.mockResolvedValueOnce({ data: mockResponse });

      const result = await client.embedText('test text');

      expect(result).toEqual(mockResponse);
      expect(axiosInstance.post).toHaveBeenCalledWith('/embed', { text: 'test text' });
    });

    it('should handle API error with details', async () => {
      const errorResponse = {
        data: {
          error: 'Invalid input',
          details: 'Text is too long',
        },
      };
      axiosInstance.post.mockRejectedValueOnce({ response: errorResponse });

      await expect(client.embedText('test text')).rejects.toThrow('Text is too long');
    });

    it('should handle API error without details', async () => {
      const errorResponse = {
        data: {
          error: 'Invalid input',
        },
      };
      axiosInstance.post.mockRejectedValueOnce({ response: errorResponse });

      await expect(client.embedText('test text')).rejects.toThrow('Invalid input');
    });
  });

  describe('embedTexts', () => {
    const mockResponse = {
      embeddings: [
        {
          embedding: [0.1, 0.2, 0.3],
          text: 'test text 1',
          vector_id: 1,
        },
        {
          embedding: [0.4, 0.5, 0.6],
          text: 'test text 2',
          vector_id: 2,
        },
      ],
    };

    it('should successfully embed multiple texts', async () => {
      axiosInstance.post.mockResolvedValueOnce({ data: mockResponse });

      const result = await client.embedTexts(['test text 1', 'test text 2']);

      expect(result).toEqual(mockResponse);
      expect(axiosInstance.post).toHaveBeenCalledWith('/embed/batch', {
        texts: ['test text 1', 'test text 2'],
      });
    });

    it('should handle batch API error', async () => {
      const errorResponse = {
        data: {
          error: 'Batch too large',
          details: 'Maximum 32 texts allowed',
        },
      };
      axiosInstance.post.mockRejectedValueOnce({ response: errorResponse });

      await expect(client.embedTexts(['text1', 'text2'])).rejects.toThrow('Maximum 32 texts allowed');
    });
  });

  describe('searchSimilar', () => {
    const mockResponse = {
      query: 'test query',
      results: [
        {
          text: 'similar text 1',
          score: 0.9,
          vector_id: 1,
        },
        {
          text: 'similar text 2',
          score: 0.8,
          vector_id: 2,
        },
      ],
    };

    it('should successfully search similar texts', async () => {
      axiosInstance.post.mockResolvedValueOnce({ data: mockResponse });

      const result = await client.searchSimilar('test query', 2);

      expect(result).toEqual(mockResponse);
      expect(axiosInstance.post).toHaveBeenCalledWith('/search', {
        text: 'test query',
        k: 2,
      });
    });

    it('should use default k value if not provided', async () => {
      axiosInstance.post.mockResolvedValueOnce({ data: mockResponse });

      await client.searchSimilar('test query');

      expect(axiosInstance.post).toHaveBeenCalledWith('/search', {
        text: 'test query',
        k: 5,
      });
    });

    it('should handle search API error', async () => {
      const errorResponse = {
        data: {
          error: 'Search failed',
          details: 'Vector store is empty',
        },
      };
      axiosInstance.post.mockRejectedValueOnce({ response: errorResponse });

      await expect(client.searchSimilar('test query')).rejects.toThrow('Vector store is empty');
    });
  });

  describe('retry mechanism', () => {
    beforeEach(() => {
      // Мокаем setTimeout для тестов с задержкой
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
      jest.clearAllMocks();
    });

    it('should retry failed requests', async () => {
      const mockResponse = {
        embedding: [0.1, 0.2, 0.3],
        text: 'test text',
        vector_id: 1,
      };

      // Настраиваем мок для последовательных вызовов
      let callCount = 0;
      axiosInstance.post.mockImplementation(() => {
        callCount++;
        if (callCount < 3) {
          return Promise.reject(new Error('Network error'));
        }
        return Promise.resolve({ data: mockResponse });
      });

      // Запускаем запрос
      const resultPromise = client.embedText('test text');
      
      // Ждем все таймеры и промисы
      await Promise.all([
        resultPromise,
        jest.runAllTimersAsync()
      ]);
      
      const result = await resultPromise;
      expect(result).toEqual(mockResponse);
      expect(axiosInstance.post).toHaveBeenCalledTimes(3);
    });

    it('should fail after max retries', async () => {
      // Настраиваем мок для возврата ошибки
      axiosInstance.post.mockRejectedValue(new Error('Network error'));

      // Запускаем запрос и ждем все таймеры
      const resultPromise = client.embedText('test text');
      
      await Promise.all([
        expect(resultPromise).rejects.toThrow('Network error'),
        jest.runAllTimersAsync()
      ]);
      
      expect(axiosInstance.post).toHaveBeenCalledTimes(3); // maxRetries = 3
    });

    it('should use exponential backoff', async () => {
      const mockResponse = {
        embedding: [0.1, 0.2, 0.3],
        text: 'test text',
        vector_id: 1,
      };

      // Настраиваем мок для последовательных вызовов
      let callCount = 0;
      axiosInstance.post.mockImplementation(() => {
        callCount++;
        if (callCount < 3) {
          return Promise.reject(new Error('Network error'));
        }
        return Promise.resolve({ data: mockResponse });
      });

      // Запускаем запрос
      const resultPromise = client.embedText('test text');

      // Первая попытка происходит сразу
      expect(axiosInstance.post).toHaveBeenCalledTimes(1);

      // Ждем первую задержку (100мс)
      await jest.advanceTimersByTimeAsync(100);
      expect(axiosInstance.post).toHaveBeenCalledTimes(2);

      // Ждем вторую задержку (200мс)
      await jest.advanceTimersByTimeAsync(200);
      expect(axiosInstance.post).toHaveBeenCalledTimes(3);

      // Завершаем все оставшиеся таймеры
      await jest.runAllTimersAsync();
      
      const result = await resultPromise;
      expect(result).toEqual(mockResponse);
    });

    it('should not retry API errors', async () => {
      const errorResponse = {
        data: {
          error: 'Invalid input',
          details: 'Text is too long',
        },
      };

      // Настраиваем мок для возврата ошибки API
      axiosInstance.post.mockRejectedValue({
        isAxiosError: true,
        response: errorResponse,
      });

      // Запускаем запрос и ждем все таймеры
      const resultPromise = client.embedText('test text');
      
      await Promise.all([
        expect(resultPromise).rejects.toThrow('Text is too long'),
        jest.runAllTimersAsync()
      ]);
      
      expect(axiosInstance.post).toHaveBeenCalledTimes(1); // Не должно быть повторных попыток
    });
  });
}); 