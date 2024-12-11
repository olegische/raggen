import { PrismaClient, Message } from '@prisma/client';
import { BaseRepository } from './base.repository';

export interface CreateMessageParams {
  chatId: string;
  message: string;
  model: string;
  provider: string;
  temperature: number;
  maxTokens: number;
  response?: string | null;
}

export interface UpdateMessageParams {
  response?: string | null;
}

export class MessageRepository extends BaseRepository {
  constructor(prisma: PrismaClient) {
    super(prisma);
  }

  async createMessage(params: CreateMessageParams): Promise<Message> {
    try {
      this.validateCreateMessageParams(params);

      return await this.prisma.message.create({
        data: params
      });
    } catch (error) {
      console.error('Error creating message:', error);
      throw error instanceof Error ? error : new Error('Failed to create message');
    }
  }

  async getMessage(id: number): Promise<Message | null> {
    try {
      if (!id || id <= 0) {
        throw new Error('Invalid message ID');
      }

      return await this.prisma.message.findUnique({
        where: { id }
      });
    } catch (error) {
      console.error('Error getting message:', error);
      throw error instanceof Error ? error : new Error('Failed to get message');
    }
  }

  async getMessagesByChat(chatId: string): Promise<Message[]> {
    try {
      if (!chatId) {
        throw new Error('Invalid chat ID');
      }

      return await this.prisma.message.findMany({
        where: { chatId },
        orderBy: { timestamp: 'asc' }
      });
    } catch (error) {
      console.error('Error getting messages by chat:', error);
      throw error instanceof Error ? error : new Error('Failed to get messages');
    }
  }

  async updateMessage(id: number, params: UpdateMessageParams): Promise<Message> {
    try {
      if (!id || id <= 0) {
        throw new Error('Invalid message ID');
      }

      return await this.prisma.message.update({
        where: { id },
        data: params
      });
    } catch (error) {
      console.error('Error updating message:', error);
      throw error instanceof Error ? error : new Error('Failed to update message');
    }
  }

  async deleteMessage(id: number): Promise<Message> {
    try {
      if (!id || id <= 0) {
        throw new Error('Invalid message ID');
      }

      return await this.prisma.message.delete({
        where: { id }
      });
    } catch (error) {
      console.error('Error deleting message:', error);
      throw error instanceof Error ? error : new Error('Failed to delete message');
    }
  }

  private validateCreateMessageParams(params: CreateMessageParams): void {
    if (!params.chatId) {
      throw new Error('Chat ID is required');
    }

    if (!params.message || params.message.trim().length === 0) {
      throw new Error('Message content is required');
    }

    if (!params.model || params.model.trim().length === 0) {
      throw new Error('Model is required');
    }

    if (!params.provider || params.provider.trim().length === 0) {
      throw new Error('Provider is required');
    }

    if (typeof params.temperature !== 'number' || params.temperature < 0 || params.temperature > 1) {
      throw new Error('Temperature must be between 0 and 1');
    }

    if (typeof params.maxTokens !== 'number' || params.maxTokens <= 0) {
      throw new Error('Max tokens must be greater than 0');
    }
  }
}
