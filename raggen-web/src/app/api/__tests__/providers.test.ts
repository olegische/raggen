import { GET } from '../providers/route';
import { ProviderService } from '@/services/provider.service';

jest.mock('@/services/provider.service', () => ({
  ProviderService: {
    getAvailableProviders: jest.fn()
  }
}));

describe('Providers API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return available providers', async () => {
    const mockProviders = ['yandex', 'gigachat'];
    (ProviderService.getAvailableProviders as jest.Mock).mockResolvedValue(mockProviders);

    const response = await GET();
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual(mockProviders);
    expect(ProviderService.getAvailableProviders).toHaveBeenCalled();
  });

  it('should handle errors', async () => {
    (ProviderService.getAvailableProviders as jest.Mock).mockRejectedValue(new Error('Test error'));

    const response = await GET();

    expect(response.status).toBe(500);
    expect(await response.text()).toBe('Test error');
  });
}); 