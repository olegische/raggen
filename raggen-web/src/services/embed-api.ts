import axios, { AxiosInstance, AxiosResponse } from 'axios';

export interface EmbeddingResponse {
  embedding: number[];
  text: string;
  vector_id: number;
}

export interface BatchEmbeddingResponse {
  embeddings: EmbeddingResponse[];
}

export interface SearchResult {
  text: string;
  score: number;
  vector_id: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
}

export interface ErrorResponse {
  error: string;
  details?: string;
}

export class EmbedApiClient {
  private client: AxiosInstance;
  private maxRetries: number;
  private retryDelay: number;

  constructor(
    baseURL: string = process.env.NEXT_PUBLIC_EMBED_API_URL || 'http://localhost:8001',
    maxRetries: number = 3,
    retryDelay: number = 1000
  ) {
    this.client = axios.create({
      baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });
    this.maxRetries = maxRetries;
    this.retryDelay = retryDelay;
  }

  private async retryRequest<T>(request: () => Promise<AxiosResponse<T>>): Promise<AxiosResponse<T>> {
    let retryCount = 0;
    let lastError: any;

    while (retryCount < this.maxRetries) {
      try {
        return await request();
      } catch (error) {
        lastError = error;
        
        // Если это ошибка API, не нужно повторять запрос
        if (axios.isAxiosError(error) && error.response?.data) {
          throw error;
        }

        retryCount++;

        if (retryCount >= this.maxRetries) {
          break;
        }

        // Экспоненциальная задержка
        const delay = this.retryDelay * Math.pow(2, retryCount - 1);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  }

  /**
   * Генерирует эмбеддинг для одного текста
   */
  async embedText(text: string): Promise<EmbeddingResponse> {
    try {
      const response = await this.retryRequest(() => 
        this.client.post<EmbeddingResponse>('/embed', { text })
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.data) {
        const errorData = error.response.data as ErrorResponse;
        throw new Error(errorData.details || errorData.error);
      }
      throw error;
    }
  }

  /**
   * Генерирует эмбеддинги для нескольких текстов
   */
  async embedTexts(texts: string[]): Promise<BatchEmbeddingResponse> {
    try {
      const response = await this.retryRequest(() => 
        this.client.post<BatchEmbeddingResponse>('/embed/batch', { texts })
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.data) {
        const errorData = error.response.data as ErrorResponse;
        throw new Error(errorData.details || errorData.error);
      }
      throw error;
    }
  }

  /**
   * Ищет похожие тексты
   */
  async searchSimilar(text: string, k: number = 5): Promise<SearchResponse> {
    try {
      const response = await this.retryRequest(() => 
        this.client.post<SearchResponse>('/search', { text, k })
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.data) {
        const errorData = error.response.data as ErrorResponse;
        throw new Error(errorData.details || errorData.error);
      }
      throw error;
    }
  }
} 