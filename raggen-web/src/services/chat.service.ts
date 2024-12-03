import { ProviderFactory, ProviderType } from '@/providers/factory';
import { GenerationOptions, ProviderMessage } from '@/providers/base.provider';
import { GENERATION_CONFIG } from '@/config/generation';
import prisma from '@/lib/db';

export class ChatService {
  private provider;

  constructor(providerType: ProviderType) {
    this.provider = ProviderFactory.createProvider(providerType);
  }

  async sendMessage(
    message: string,
    chatId?: string,
    options?: GenerationOptions
  ) {
    try {
      // Получаем или создаем чат
      const chat = chatId 
        ? await prisma.chat.findUnique({ where: { id: chatId } })
        : await prisma.chat.create({ 
            data: { 
              provider: this.provider.constructor.name.replace('Provider', '').toLowerCase()
            } 
          });

      if (!chat) {
        throw new Error('Chat not found');
      }

      // Получаем предыдущие сообщения для контекста
      const previousMessages = await prisma.message.findMany({
        where: { chatId: chat.id },
        orderBy: { timestamp: 'asc' },
        take: 10
      });

      // Форматируем сообщения для провайдера
      const context = previousMessages.map(msg => ({
        role: msg.response ? 'assistant' : 'user',
        content: msg.response || msg.message
      })) as ProviderMessage[];

      // Получаем ответ от провайдера
      const response = await this.provider.generateResponse(
        message,
        options,
        context
      );

      // Сохраняем сообщение в БД
      const savedMessage = await prisma.message.create({
        data: {
          chatId: chat.id,
          message: message,
          response: response.text,
          model: options?.model || 'default',
          provider: this.provider.constructor.name.replace('Provider', '').toLowerCase(),
          temperature: options?.temperature || GENERATION_CONFIG.temperature.default,
          maxTokens: options?.maxTokens || GENERATION_CONFIG.maxTokens.default
        }
      });

      return {
        message: savedMessage,
        chatId: chat.id,
        usage: response.usage
      };

    } catch (error) {
      console.error('Error in ChatService:', error);
      throw error;
    }
  }

  async getHistory(chatId: string) {
    return prisma.message.findMany({
      where: { chatId },
      orderBy: { timestamp: 'asc' }
    });
  }
} 