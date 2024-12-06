import { render, screen } from '@testing-library/react';
import ChatWindow from '../ChatWindow';
import { Message } from '@prisma/client';
import { ContextSearchResult } from '@/services/context.service';

describe('ChatWindow', () => {
  const mockMessage: Message = {
    id: 1,
    chatId: 'chat-1',
    message: 'Test message',
    response: 'Test response',
    model: 'test-model',
    provider: 'yandex',
    temperature: 0.7,
    maxTokens: 1000,
    timestamp: new Date()
  };

  const mockContext: ContextSearchResult[] = [
    {
      message: {
        id: 2,
        chatId: 'chat-1',
        message: 'Previous relevant message',
        model: 'test-model',
        provider: 'test-provider',
        temperature: 0.7,
        maxTokens: 1000,
        timestamp: new Date(),
        response: null
      },
      score: 0.85,
      usedInPrompt: true
    }
  ];

  it('renders messages with context correctly', () => {
    const messageContexts = {
      [mockMessage.id]: mockContext
    };

    render(
      <ChatWindow
        messages={[mockMessage]}
        provider="yandex"
        messageContexts={messageContexts}
      />
    );

    expect(screen.getByText('Test message')).toBeInTheDocument();
    expect(screen.getByText('Test response')).toBeInTheDocument();
    expect(screen.getByText('Использованный контекст:')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('Previous relevant message')).toBeInTheDocument();
  });

  it('renders messages without context correctly', () => {
    render(
      <ChatWindow
        messages={[mockMessage]}
        provider="yandex"
      />
    );

    expect(screen.getByText('Test message')).toBeInTheDocument();
    expect(screen.getByText('Test response')).toBeInTheDocument();
    expect(screen.queryByText('Использованный контекст:')).not.toBeInTheDocument();
  });

  it('renders empty state when no messages', () => {
    render(<ChatWindow messages={[]} provider="yandex" />);
    expect(screen.getByText('Начните диалог, отправив сообщение')).toBeInTheDocument();
  });

  it('renders error state', () => {
    render(
      <ChatWindow
        messages={[]}
        provider="yandex"
        error="Test error"
      />
    );
    expect(screen.getByText('Test error')).toBeInTheDocument();
  });

  it('renders loading state', () => {
    render(
      <ChatWindow
        messages={[]}
        provider="yandex"
        loading={true}
      />
    );
    expect(screen.getByText(/печатает/)).toBeInTheDocument();
  });
}); 