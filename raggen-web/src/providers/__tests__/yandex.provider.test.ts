import { YandexGPTProvider } from '../yandex/provider';
import { ProviderConfig, ProviderMessage } from '../base.provider';

describe('YandexGPTProvider', () => {
  let provider: YandexGPTProvider;
  const mockConfig: ProviderConfig = {
    apiUrl: 'https://yandex.api',
    credentials: 'yandex-key'
  };

  beforeEach(() => {
    provider = new YandexGPTProvider(mockConfig);
    global.fetch = jest.fn();
  });

  describe('generateResponse', () => {
    const mockMessage = 'Test message';
    const mockResponse = {
      result: {
        alternatives: [{
          message: {
            role: 'assistant',
            text: 'Test response'
          }
        }],
        usage: {
          inputTextTokens: '10',
          completionTokens: '20',
          totalTokens: '30'
        }
      }
    };

    beforeEach(() => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });
    });

    it('should generate response with correct configuration', async () => {
      const result = await provider.generateResponse(mockMessage);

      expect(global.fetch).toHaveBeenCalledWith(
        mockConfig.apiUrl,
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Api-Key ${mockConfig.credentials}`,
            'x-folder-id': process.env.YANDEX_FOLDER_ID
          }
        })
      );

      expect(result).toEqual({
        text: 'Test response',
        usage: {
          promptTokens: 10,
          completionTokens: 20,
          totalTokens: 30
        }
      });
    });

    it('should handle previous messages correctly', async () => {
      const previousMessages: ProviderMessage[] = [
        { role: 'user', content: 'Previous message' }
      ];

      await provider.generateResponse(mockMessage, undefined, previousMessages);

      const requestBody = JSON.parse(
        (global.fetch as jest.Mock).mock.calls[0][1].body
      );

      expect(requestBody.messages).toEqual([
        ...previousMessages,
        { role: 'user', content: mockMessage }
      ]);
    });

    it('should handle API errors', async () => {
      (global.fetch as jest.Mock).mockResolvedValue({
        ok: false,
        status: 400
      });

      await expect(provider.generateResponse(mockMessage))
        .rejects
        .toThrow('YandexGPT API error: 400');
    });
  });

  describe('listModels', () => {
    it('should return available models', async () => {
      const models = await provider.listModels();
      
      expect(models).toEqual([
        'YandexGPT Lite Latest',
        'YandexGPT Lite RC',
        'YandexGPT Pro Latest',
        'YandexGPT Pro RC',
        'YandexGPT Pro 32k RC'
      ]);
    });
  });
}); 