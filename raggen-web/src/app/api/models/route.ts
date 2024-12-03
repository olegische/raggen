import { NextRequest } from 'next/server';
import { ModelService } from '@/services/model.service';
import { ProviderType } from '@/providers/factory';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const provider = searchParams.get('provider');

    if (!provider) {
      return new Response('Provider is required', { status: 400 });
    }

    const models = await ModelService.getModels(provider as ProviderType);
    return Response.json(models);

  } catch (error) {
    console.error('Error in models API:', error);
    return new Response(
      error instanceof Error ? error.message : 'Internal Server Error',
      { status: 500 }
    );
  }
} 