import { ProviderService } from '../../../services/provider.service';
import { ProviderType } from '../../../config/providers';

export async function GET() {
  try {
    const providers = await ProviderService.getAvailableProviders();
    const providerStatuses = providers.map((provider: ProviderType) => ({
      id: provider,
      status: ProviderService.getProviderStatus(provider)
    }));

    return Response.json(providerStatuses);

  } catch (error) {
    console.error('Error in providers API:', error);
    return new Response(
      error instanceof Error ? error.message : 'Internal Server Error',
      { status: 500 }
    );
  }
}
