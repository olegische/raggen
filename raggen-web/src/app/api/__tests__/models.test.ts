import { GET } from '../models/route';
import { ModelService } from '@/services/model.service';
import { NextRequest } from 'next/server';

jest.mock('@/services/model.service');

describe('Models API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return models for provider', async () => {
    const mockModels = ['model1', 'model2'];
    (ModelService.getModels as jest.Mock).mockResolvedValue(mockModels);

    const request = new NextRequest(
      'http://localhost/api/models?provider=yandex'
    );

    const response = await GET(request);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data).toEqual(mockModels);
    expect(ModelService.getModels).toHaveBeenCalledWith('yandex');
  });

  it('should require provider parameter', async () => {
    const request = new NextRequest('http://localhost/api/models');

    const response = await GET(request);
    expect(response.status).toBe(400);
  });

  it('should handle errors', async () => {
    (ModelService.getModels as jest.Mock).mockRejectedValue(
      new Error('Test error')
    );

    const request = new NextRequest(
      'http://localhost/api/models?provider=yandex'
    );

    const response = await GET(request);
    expect(response.status).toBe(500);
  });
}); 