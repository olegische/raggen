import { POST } from '../chat/route';
import { ChatService } from '@/services/chat.service';
import { NextRequest } from 'next/server';

jest.mock('@/services/chat.service');

describe('Chat API', () => {
  const mockChatService = {
    sendMessage: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (ChatService as jest.Mock).mockImplementation(() => mockChatService);
  });

  it('should handle message generation', async () => {
    const mockResult = {
      message: { id: 1, text: 'Test response' },
      chatId: 'test-chat',
      usage: { totalTokens: 100 }
    };

    mockChatService.sendMessage.mockResolvedValue(mockResult);

    const request = new NextRequest('http://localhost/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        message: 'Test message',
        provider: 'yandex',
        options: {
          model: 'test-model',
          temperature: 0.7
        }
      })
    });

    const response = await POST(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual(mockResult);
    expect(mockChatService.sendMessage).toHaveBeenCalledWith(
      'Test message',
      undefined,
      expect.objectContaining({
        model: 'test-model',
        temperature: 0.7
      })
    );
  });

  it('should require message', async () => {
    const request = new NextRequest('http://localhost/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        provider: 'yandex'
      })
    });

    const response = await POST(request);
    expect(response.status).toBe(400);
  });

  it('should require provider', async () => {
    const request = new NextRequest('http://localhost/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        message: 'Test'
      })
    });

    const response = await POST(request);
    expect(response.status).toBe(400);
  });

  it('should handle errors', async () => {
    mockChatService.sendMessage.mockRejectedValue(new Error('Test error'));

    const request = new NextRequest('http://localhost/api/chat', {
      method: 'POST',
      body: JSON.stringify({
        message: 'Test',
        provider: 'yandex'
      })
    });

    const response = await POST(request);
    expect(response.status).toBe(500);
  });
}); 