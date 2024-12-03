import { ChatService } from '../chat.service';
import { ProviderFactory } from '@/providers/factory';
import prisma from '@/lib/db';

jest.mock('@/providers/factory');
jest.mock('@/lib/db');

describe('ChatService', () => {
  let chatService: ChatService;
  
  beforeEach(() => {
    chatService = new ChatService('yandex');
  });

  it('should be defined', () => {
    expect(chatService).toBeDefined();
  });
});