import { v4 as uuidv4 } from 'uuid';
import { z } from 'zod';
import axios from 'axios';
import https from 'https';
import { BaseProvider, ProviderConfig, ProviderMessage, GenerationOptions, GenerationResult } from '../base.provider';

const responseSchema = z.object({
  choices: z.array(
    z.object({
      message: z.object({
        role: z.string(),
        content: z.string()
      })
    })
  ),
  usage: z.object({
    prompt_tokens: z.number(),
    completion_tokens: z.number(),
    total_tokens: z.number()
  }).optional()
});

type GigaChatResponse = z.infer<typeof responseSchema>;

export class GigaChatProvider extends BaseProvider {
  private accessToken?: string;
  private tokenExpiration?: number;

  constructor(config: ProviderConfig) {
    super(config, responseSchema);
  }

  async generateResponse(
    message: string,
    options?: GenerationOptions,
    previousMessages?: ProviderMessage[]
  ): Promise<GenerationResult> {
    const validatedOptions = this.validateOptions(options);
    await this.ensureValidToken();

    try {
      console.log('Generating response with options:', validatedOptions);
      
      const messages = this.formatMessages(message, previousMessages);
      console.log('Formatted messages:', messages);

      const response = await axios.post(
        `${this.config.apiUrl}/chat/completions`,
        {
          model: 'GigaChat',
          messages: messages,
          temperature: validatedOptions.temperature,
          max_tokens: validatedOptions.maxTokens
        },
        {
          headers: {
            'Authorization': `Bearer ${this.accessToken}`,
            'Content-Type': 'application/json'
          },
          httpsAgent: new https.Agent({ rejectUnauthorized: false })
        }
      );

      console.log('Raw API response:', JSON.stringify(response.data, null, 2));
      const validated = this.validateResponse(response.data) as GigaChatResponse;

      return {
        text: validated.choices[0].message.content,
        usage: validated.usage ? {
          promptTokens: validated.usage.prompt_tokens,
          completionTokens: validated.usage.completion_tokens,
          totalTokens: validated.usage.total_tokens
        } : undefined
      };
    } catch (error) {
      console.error('Error in GigaChatProvider:', error);
      throw this.formatError(error);
    }
  }

  async listModels(): Promise<string[]> {
    try {
      console.log('Ensuring valid token for GigaChat');
      await this.ensureValidToken();

      console.log('Requesting GigaChat models');
      const response = await axios.get(
        `${this.config.apiUrl}/models`,
        {
          headers: {
            'Authorization': `Bearer ${this.accessToken}`
          },
          httpsAgent: new https.Agent({ rejectUnauthorized: false })
        }
      );

      const models = response.data.data.map((model: any) => model.id);
      console.log('Available GigaChat models:', models);
      return models;
    } catch (error) {
      console.error('Error listing GigaChat models:', error);
      throw this.formatError(error);
    }
  }

  private async ensureValidToken(): Promise<void> {
    if (this.accessToken && this.tokenExpiration && Date.now() < this.tokenExpiration) {
      console.log('Using existing GigaChat token');
      return;
    }

    try {
      console.log('Requesting new GigaChat token');
      const response = await axios.post(
        'https://ngw.devices.sberbank.ru:9443/api/v2/oauth',
        'scope=GIGACHAT_API_PERS',
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json',
            'RqUID': uuidv4(),
            'Authorization': `Basic ${this.config.credentials}`
          },
          httpsAgent: new https.Agent({ rejectUnauthorized: false })
        }
      );

      this.accessToken = response.data.access_token;
      this.tokenExpiration = Date.now() + 29 * 60 * 1000; // 29 minutes
      console.log('Successfully obtained new GigaChat token');
    } catch (error) {
      if (axios.isAxiosError(error)) {
        console.error('GigaChat token error details:', {
          status: error.response?.status,
          statusText: error.response?.statusText,
          data: error.response?.data,
          message: error.message
        });
      }
      throw this.formatError(error);
    }
  }

  protected formatError(error: unknown): Error {
    if (error instanceof Error) {
      return error;
    }
    return new Error('Unknown error occurred in GigaChat provider');
  }
} 