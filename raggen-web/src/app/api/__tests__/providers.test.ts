import { GET } from '../providers/route';
import { ProviderService } from '@/services/provider.service';
import { NextRequest } from 'next/server';

jest.mock('@/services/provider.service');

describe('Providers API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return available providers with status', async () => {
    const mockProviders = ['yandex', 'gigachat'];
    const mockStatus = {
      available: true,
      lastCheck: Date.now()
    };

    (ProviderService.getAvailableProviders as jest.Mock).mockResolvedValue(mockProviders);
    (ProviderService.getProviderStatus as jest.Mock).mockReturnValue(mockStatus);

    const request = new NextRequest('http://localhost/api/providers');
    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual([
      { id: 'yandex', status: mockStatus },
      { id: 'gigachat', status: mockStatus }
    ]);
  });

  it('should handle errors', async () => {
    (ProviderService.getAvailableProviders as jest.Mock).mockRejectedValue(
      new Error('Test error')
    );

    const request = new NextRequest('http://localhost/api/providers');
    const response = await GET(request);

    expect(response.status).toBe(500);
  });
}); 