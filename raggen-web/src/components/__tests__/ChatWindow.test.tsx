import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from 'next-themes';
import ChatWindow from '../ChatWindow';
import { Message } from '@prisma/client';
import { getProviderDisplayName } from '@/config/providers';

// Mock clipboard API
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn()
  }
});

describe('ChatWindow', () => {
  const mockMessages = [{
    id: 1,
    chatId: 'test-chat',
    message: 'Hello',
    response: 'Hi there!',
    model: 'test-model',
    provider: 'yandex',
    temperature: 0.3,
    maxTokens: 2000,
    timestamp: new Date()
  }] as Message[];

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render empty state', () => {
    render(
      <ThemeProvider>
        <ChatWindow messages={[]} provider="yandex" />
      </ThemeProvider>
    );

    expect(screen.getByText('Начните диалог, отправив сообщение')).toBeInTheDocument();
  });

  it('should render messages', () => {
    render(
      <ThemeProvider>
        <ChatWindow messages={mockMessages} provider="yandex" />
      </ThemeProvider>
    );

    expect(screen.getByText('Hello')).toBeInTheDocument();
    expect(screen.getByText('Hi there!')).toBeInTheDocument();
    expect(screen.getByText('Модель: test-model')).toBeInTheDocument();
  });

  it('should copy user message to clipboard', async () => {
    render(
      <ThemeProvider>
        <ChatWindow messages={mockMessages} provider="yandex" />
      </ThemeProvider>
    );

    const copyButtons = screen.getAllByTitle('Копировать текст');
    fireEvent.click(copyButtons[0]); // User message copy button

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('Hello');
  });

  it('should copy provider response to clipboard', async () => {
    render(
      <ThemeProvider>
        <ChatWindow messages={mockMessages} provider="yandex" />
      </ThemeProvider>
    );

    const copyButtons = screen.getAllByTitle('Копировать текст');
    fireEvent.click(copyButtons[1]); // Provider response copy button is still second in the list

    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('Hi there!');
  });

  it('should show loading state', () => {
    render(
      <ThemeProvider>
        <ChatWindow messages={[]} provider="yandex" loading={true} />
      </ThemeProvider>
    );

    expect(screen.getByText(`${getProviderDisplayName('yandex')} печатает...`)).toBeInTheDocument();
  });

  it('should show error state', () => {
    const error = 'Test error message';
    render(
      <ThemeProvider>
        <ChatWindow messages={[]} provider="yandex" error={error} />
      </ThemeProvider>
    );

    expect(screen.getByText(error)).toBeInTheDocument();
  });

  it('should show provider name for responses', () => {
    render(
      <ThemeProvider>
        <ChatWindow messages={mockMessages} provider="yandex" />
      </ThemeProvider>
    );

    expect(screen.getByText(getProviderDisplayName('yandex'))).toBeInTheDocument();
    expect(screen.getByText('Вы')).toBeInTheDocument();
  });

  it('should show copy buttons in correct positions', () => {
    render(
      <ThemeProvider>
        <ChatWindow messages={mockMessages} provider="yandex" />
      </ThemeProvider>
    );

    const userMessage = screen.getByText('Hello').closest('.group');
    const providerResponse = screen.getByText('Hi there!').closest('.group');

    expect(userMessage?.querySelector('button[title="Копировать текст"]')).toBeInTheDocument();
    expect(providerResponse?.querySelector('button[title="Копировать текст"]')).toBeInTheDocument();
  });
}); 