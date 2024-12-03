import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ThemeProvider } from 'next-themes';
import ProviderSelector from '../ProviderSelector';

describe('ProviderSelector', () => {
  const mockProviders = [
    { id: 'yandex', status: { available: true, lastCheck: Date.now() } },
    { id: 'gigachat', status: { available: true, lastCheck: Date.now() } }
  ];

  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it('should render provider selector', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockProviders
    });

    const onProviderChange = jest.fn();
    render(
      <ThemeProvider>
        <ProviderSelector
          selectedProvider="yandex"
          onProviderChange={onProviderChange}
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByLabelText('Провайдер:')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    expect(select).toHaveValue('yandex');
  });

  it('should handle provider change', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockProviders
    });

    const onProviderChange = jest.fn();
    render(
      <ThemeProvider>
        <ProviderSelector
          selectedProvider="yandex"
          onProviderChange={onProviderChange}
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'gigachat' }
    });

    expect(onProviderChange).toHaveBeenCalledWith('gigachat');
  });

  it('should show loading state', () => {
    (global.fetch as jest.Mock).mockImplementationOnce(
      () => new Promise(() => {}) // Never resolves
    );

    render(
      <ThemeProvider>
        <ProviderSelector
          selectedProvider="yandex"
          onProviderChange={jest.fn()}
        />
      </ThemeProvider>
    );

    expect(screen.getByText('Загрузка провайдеров...')).toBeInTheDocument();
  });

  it('should show error state', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Test error'));

    render(
      <ThemeProvider>
        <ProviderSelector
          selectedProvider="yandex"
          onProviderChange={jest.fn()}
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Ошибка: Test error')).toBeInTheDocument();
    });
  });

  it('should disable selector when specified', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockProviders
    });

    render(
      <ThemeProvider>
        <ProviderSelector
          selectedProvider="yandex"
          onProviderChange={jest.fn()}
          disabled
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeDisabled();
    });
  });
}); 