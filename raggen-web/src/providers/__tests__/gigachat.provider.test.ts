import { GigaChatProvider } from '../gigachat/provider';
import { ProviderConfig, ProviderMessage } from '../base.provider';
import axios from 'axios';
import https from 'https';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

interface CompletionBody {
  messages: ProviderMessage[];
  model: string;
  temperature: number;
  max_tokens: number;
}

describe('GigaChatProvider', () => {
  let provider: GigaChatProvider;
  const mockConfig: ProviderConfig = {
    apiUrl: 'https://gigachat.api',
    credentials: 'gigachat-credentials'
  };

  const expectedAxiosConfig = {
    headers: {
      'Content-Type': 'application/json',
      'Authorization': expect.stringMatching(/^Bearer .+$/),
    },
    httpsAgent: expect.any(https.Agent)
  };

  beforeEach(() => {
    provider = new GigaChatProvider(mockConfig);
    jest.clearAllMocks();

    // Мокаем все вызовы axios.post по умолчанию
    mockedAxios.post.mockImplementation(async (url) => {
      if (url.includes('oauth')) {
        return {
          data: {
            access_token: 'test-token',
            expires_at: Date.now() + 1000000
          }
        };
      }
      return {
        data: {
          choices: [{
            message: {
              role: 'assistant',
              content: 'Test response'
            }
          }],
          usage: {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30
          }
        }
      };
    });

    // Мокаем все вызовы axios.get по умолчанию
    mockedAxios.get.mockImplementation(async () => ({
      data: {
        data: [
          { id: 'GigaChat' },
          { id: 'GigaChat-Pro' }
        ]
      }
    }));
  });

  describe('generateResponse', () => {
    const mockMessage = 'Test message';

    it('should generate response with correct configuration', async () => {
      const result = await provider.generateResponse(mockMessage);

      // Проверяем запрос токена
      expect(mockedAxios.post).toHaveBeenCalledWith(
        'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
        'scope=GIGACHAT_API_PERS',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': `Basic ${mockConfig.credentials}`,
            'Accept': 'application/json',
            'RqUID': expect.any(String)
          }),
          httpsAgent: expect.any(https.Agent)
        })
      );

      // Проверяем запрос к API
      expect(mockedAxios.post).toHaveBeenCalledWith(
        `${mockConfig.apiUrl}/chat/completions`,
        {
          model: 'GigaChat',
          messages: [{ role: 'user', content: mockMessage }],
          temperature: 0.7,
          max_tokens: 1000
        },
        expectedAxiosConfig
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

      // Проверяем только второй вызов (первый - для получения токена)
      const [, completionCall] = mockedAxios.post.mock.calls;
      const [, completionBody] = completionCall as [string, CompletionBody];

      expect(completionBody.messages).toEqual([
        ...previousMessages,
        { role: 'user', content: mockMessage }
      ]);
    });

    it('should reuse valid token', async () => {
      await provider.generateResponse(mockMessage);
      await provider.generateResponse(mockMessage);

      // Проверяем, что запрос токена был только один раз
      const oauthCalls = mockedAxios.post.mock.calls.filter(call => 
        call[0].includes('oauth')
      );
      expect(oauthCalls).toHaveLength(1);
    });

    it('should handle API errors', async () => {
      mockedAxios.post.mockRejectedValueOnce(new Error('API Error'));

      await expect(provider.generateResponse(mockMessage))
        .rejects
        .toThrow('API Error');
    });
  });

  describe('listModels', () => {
    it('should return available models', async () => {
      const models = await provider.listModels();
      
      expect(models).toEqual(['GigaChat', 'GigaChat-Pro']);
      expect(mockedAxios.get).toHaveBeenCalledWith(
        `${mockConfig.apiUrl}/models`,
        expect.objectContaining({
          headers: {
            'Authorization': expect.stringMatching(/^Bearer .+$/),
          },
          httpsAgent: expect.any(https.Agent)
        })
      );
    });

    it('should handle API errors', async () => {
      mockedAxios.get.mockRejectedValueOnce(new Error('API Error'));

      await expect(provider.listModels())
        .rejects
        .toThrow('API Error');
    });
  });
}); 