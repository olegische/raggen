import { NextRequest } from 'next/server';
import { ChatService } from '@/services/chat.service';
import { GenerationOptions } from '@/providers/base.provider';

export async function POST(request: NextRequest) {
  try {
    const { message, chatId, provider, options } = await request.json();

    if (!message) {
      return new Response('Message is required', { status: 400 });
    }

    if (!provider) {
      return new Response('Provider is required', { status: 400 });
    }

    const chatService = new ChatService(provider);
    const generationOptions: GenerationOptions = {
      model: options?.model,
      temperature: options?.temperature,
      maxTokens: options?.maxTokens
    };

    const result = await chatService.sendMessage(
      message,
      chatId,
      generationOptions
    );

    return Response.json(result);

  } catch (error) {
    console.error('Error in chat API:', error);
    return new Response(
      error instanceof Error ? error.message : 'Internal Server Error',
      { status: 500 }
    );
  }
} 