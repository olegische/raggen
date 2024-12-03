import { GET } from '../messages/route';
import { ChatService } from '@/services/chat.service';
import { ProviderService } from '@/services/provider.service';
import { NextRequest } from 'next/server';

jest.mock('@/services/chat.service');
jest.mock('@/services/provider.service');

describe('Messages API', () => {
  const mockChatService = {
    getHistory: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (ChatService as jest.Mock).mockImplementation(() => mockChatService);
    (ProviderService.isProviderAvailable as jest.Mock).mockResolvedValue(true);
  });

  it('should return chat history', async () => {
    const mockMessages = [
      { id: 1, text: 'Test message' }
    ];

    mockChatService.getHistory.mockResolvedValue(mockMessages);

    const request = new NextRequest(
      'http://localhost/api/messages?chatId=test-chat&provider=yandex'
    );

    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual(mockMessages);
    expect(mockChatService.getHistory).toHaveBeenCalledWith('test-chat');
  });

  it('should require chatId', async () => {
    const request = new NextRequest(
      'http://localhost/api/messages?provider=yandex'
    );

    const response = await GET(request);
    expect(response.status).toBe(400);
  });

  it('should require provider', async () => {
    const request = new NextRequest(
      'http://localhost/api/messages?chatId=test-chat'
    );

    const response = await GET(request);
    expect(response.status).toBe(400);
  });

  it('should check provider availability', async () => {
    (ProviderService.isProviderAvailable as jest.Mock).mockResolvedValue(false);

    const request = new NextRequest(
      'http://localhost/api/messages?chatId=test-chat&provider=yandex'
    );

    const response = await GET(request);
    expect(response.status).toBe(503);
  });

  it('should handle errors', async () => {
    mockChatService.getHistory.mockRejectedValue(new Error('Test error'));

    const request = new NextRequest(
      'http://localhost/api/messages?chatId=test-chat&provider=yandex'
    );

    const response = await GET(request);
    expect(response.status).toBe(500);
  });
}); 