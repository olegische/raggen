import { ProviderService } from '../../../services/provider.service';

export async function GET() {
  try {
    const providerStatuses = await ProviderService.getAvailableProviders();
    return Response.json(providerStatuses);
  } catch (error) {
    console.error('Error in providers API:', error);
    return new Response(
      error instanceof Error ? error.message : 'Internal Server Error',
      { status: 500 }
    );
  }
}
