import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ThemeProvider } from 'next-themes';
import Home from '../page';

// Mock Date.now() to return a consistent value
const mockDateNow = 12345;
Date.now = jest.fn(() => mockDateNow);

describe('Home', () => {
  const mockProviders = [
    { id: 'yandex', status: { available: true } },
    { id: 'gigachat', status: { available: true } }
  ];

  beforeEach(() => {
    global.fetch = jest.fn()
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => mockProviders
      }));
  });

  it('should load default model on provider change', async () => {
    (global.fetch as jest.Mock)
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => mockProviders
      }))
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => ['model1', 'model2']
      }));

    render(
      <ThemeProvider>
        <Home />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(expect.stringContaining('/api/models?provider=yandex'));
    });
  });

  it('should handle message sending with button click', async () => {
    const testMessage = 'test message';
    const testResponse = 'test response';
    
    const mockMessage = {
      id: mockDateNow,
      chatId: 'test-chat',
      message: testMessage,
      response: testResponse,
      model: 'model1',
      provider: 'yandex',
      temperature: 0.7,
      maxTokens: 1000,
      timestamp: new Date()
    };

    (global.fetch as jest.Mock)
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => mockProviders
      }))
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => ['model1']
      }))
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => mockMessage
      }));

    render(
      <ThemeProvider>
        <Home />
      </ThemeProvider>
    );

    // Wait for the initial load
    await waitFor(() => {
      expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    // Type a message
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: testMessage } });

    // Click the send button
    const sendButton = screen.getByRole('button', { name: /отправить/i });
    fireEvent.click(sendButton);

    // Wait for the message to appear
    await waitFor(() => {
      const messageDiv = screen.getByTestId('user-message');
      expect(messageDiv).toHaveTextContent(testMessage);
    });

    // Wait for the response to appear
    await waitFor(() => {
      const responseDiv = screen.getByTestId('provider-response');
      expect(responseDiv).toHaveTextContent(testResponse);
    });
  });

  it('should handle message sending with Enter key', async () => {
    const testMessage = 'test message';
    const testResponse = 'test response';
    
    const mockMessage = {
      id: mockDateNow,
      chatId: 'test-chat',
      message: testMessage,
      response: testResponse,
      model: 'model1',
      provider: 'yandex',
      temperature: 0.7,
      maxTokens: 1000,
      timestamp: new Date()
    };

    (global.fetch as jest.Mock)
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => mockProviders
      }))
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => ['model1']
      }))
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => mockMessage
      }));

    render(
      <ThemeProvider>
        <Home />
      </ThemeProvider>
    );

    // Wait for the initial load
    await waitFor(() => {
      expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    // Type a message
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: testMessage } });

    // Press Enter
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });

    // Wait for the message to appear
    await waitFor(() => {
      const messageDiv = screen.getByTestId('user-message');
      expect(messageDiv).toHaveTextContent(testMessage);
    });

    // Wait for the response to appear
    await waitFor(() => {
      const responseDiv = screen.getByTestId('provider-response');
      expect(responseDiv).toHaveTextContent(testResponse);
    });
  });

  it('should not send message with Shift+Enter', async () => {
    render(
      <ThemeProvider>
        <Home />
      </ThemeProvider>
    );

    // Wait for the component to load
    const input = await screen.findByPlaceholderText('Введите сообщение...');

    // Type a message
    fireEvent.change(input, { target: { value: 'test message' } });
    
    // Try to send with Shift+Enter
    fireEvent.keyDown(input, { key: 'Enter', shiftKey: true });

    // Message should still be in input
    expect(input).toHaveValue('test message');
    expect(input).not.toBeDisabled();
  });

  it('should handle errors', async () => {
    const testMessage = 'test message';
    const errorMessage = 'Failed to send message';

    (global.fetch as jest.Mock)
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => mockProviders
      }))
      .mockImplementationOnce(() => Promise.resolve({
        ok: true,
        json: async () => ['model1']
      }))
      .mockImplementationOnce(() => Promise.reject(new Error(errorMessage)));

    render(
      <ThemeProvider>
        <Home />
      </ThemeProvider>
    );

    // Wait for the initial load
    await waitFor(() => {
      expect(screen.getByRole('textbox')).toBeInTheDocument();
    });

    // Type a message
    const input = screen.getByRole('textbox');
    fireEvent.change(input, { target: { value: testMessage } });

    // Click the send button
    const sendButton = screen.getByRole('button', { name: /отправить/i });
    fireEvent.click(sendButton);

    // Wait for the error message to appear
    await waitFor(() => {
      const errorDiv = screen.getByTestId('error-message');
      expect(errorDiv).toHaveTextContent(errorMessage);
      expect(errorDiv).toHaveClass('text-red-600');
    });
  });
});