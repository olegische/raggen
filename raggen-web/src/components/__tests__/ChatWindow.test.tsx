import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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

  const defaultProps = {
    messages: [mockMessage],
    provider: 'yandex' as const,
    messageContexts: {
      [mockMessage.id]: mockContext
    },
    contextSettings: {
      maxContextMessages: 5,
      contextScoreThreshold: 0.7,
      contextEnabled: true
    },
    onContextSettingsChange: jest.fn()
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders messages with context correctly', () => {
    render(<ChatWindow {...defaultProps} />);

    expect(screen.getByText('Test message')).toBeInTheDocument();
    expect(screen.getByText('Test response')).toBeInTheDocument();
    expect(screen.getAllByText('Использованный контекст:')[0]).toBeInTheDocument();
    expect(screen.getAllByText(/85%/)[0]).toBeInTheDocument();
    expect(screen.getAllByText('Previous relevant message')[0]).toBeInTheDocument();
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

  // Тсты для настроек контекста
  it('shows context settings when settings button is clicked', async () => {
    render(<ChatWindow {...defaultProps} />);
    
    const settingsButton = screen.getByLabelText('Показать настройки контекста');
    await userEvent.click(settingsButton);
    
    expect(screen.getByText('Настройки контекста')).toBeInTheDocument();
    expect(screen.getByLabelText('Максимальное количество сообщений для контекста')).toBeInTheDocument();
    expect(screen.getByLabelText('Минимальный порог релевантности контекста')).toBeInTheDocument();
  });

  it('hides context settings when settings button is clicked again', async () => {
    render(<ChatWindow {...defaultProps} />);
    
    const settingsButton = screen.getByLabelText('Показать настройки контекста');
    await userEvent.click(settingsButton);
    await userEvent.click(settingsButton);
    
    expect(screen.queryByText('Настройки контекста')).not.toBeInTheDocument();
  });

  it('calls onContextSettingsChange when settings are changed', async () => {
    render(<ChatWindow {...defaultProps} />);
    
    // Открываем настройки
    const settingsButton = screen.getByLabelText('Показать настройки контекста');
    await userEvent.click(settingsButton);
    
    // Меняем значение слайдера
    const slider = screen.getByLabelText('Максимальное количество сообщений для контекста');
    fireEvent.change(slider, { target: { value: '3' } });
    
    expect(defaultProps.onContextSettingsChange).toHaveBeenCalledWith({
      maxContextMessages: 3,
      contextScoreThreshold: 0.7,
      contextEnabled: true,
    });
  });

  // Тесты доступности
  it('has accessible settings button', () => {
    render(<ChatWindow {...defaultProps} />);
    
    const settingsButton = screen.getByLabelText('Показать настройки контекста');
    expect(settingsButton).toHaveAttribute('aria-label');
  });

  it('supports keyboard navigation in settings panel', async () => {
    render(<ChatWindow {...defaultProps} />);
    
    // Открываем настройки
    const settingsButton = screen.getByLabelText('Показать настройки контекста');
    await userEvent.click(settingsButton);
    
    // Проверяем, что все элементы управления доступны с клавиатуры
    const checkbox = screen.getByRole('checkbox');
    const sliders = screen.getAllByRole('slider');
    
    checkbox.focus();
    expect(document.activeElement).toBe(checkbox);
    
    sliders.forEach(slider => {
      slider.focus();
      expect(document.activeElement).toBe(slider);
    });
  });
}); 