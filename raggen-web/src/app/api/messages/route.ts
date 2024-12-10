import { NextRequest } from 'next/server';
import { ChatService } from '../../../services/chat.service';
import { ProviderService } from '../../../services/provider.service';
import { ProviderType } from '../../../config/providers';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const chatId = searchParams.get('chatId');
    const provider = searchParams.get('provider') as ProviderType;

    if (!chatId) {
      return new Response('Chat ID is required', { status: 400 });
    }

    if (!provider) {
      return new Response('Provider is required', { status: 400 });
    }

    // Проверяем доступность провайдера
    const isAvailable = await ProviderService.isProviderAvailable(provider);
    if (!isAvailable) {
      return new Response('Provider is not available', { status: 503 });
    }

    const chatService = new ChatService(provider);
    const messages = await chatService.getHistory(chatId);

    return Response.json(messages);

  } catch (error) {
    console.error('Error in messages API:', error);
    return new Response(
      error instanceof Error ? error.message : 'Internal Server Error',
      { status: 500 }
    );
  }
}

export const dynamic = 'force-dynamic';
