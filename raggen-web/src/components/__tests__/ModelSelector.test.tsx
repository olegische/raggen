import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ThemeProvider } from 'next-themes';
import ModelSelector from '../ModelSelector';

describe('ModelSelector', () => {
  const mockModels = ['model1', 'model2'];

  beforeEach(() => {
    global.fetch = jest.fn();
  });

  it('should render model selector', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockModels
    });

    const onModelChange = jest.fn();
    render(
      <ThemeProvider>
        <ModelSelector
          selectedModel="model1"
          provider="yandex"
          onModelChange={onModelChange}
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByLabelText('Модель:')).toBeInTheDocument();
    });

    const select = screen.getByRole('combobox');
    expect(select).toHaveValue('model1');
  });

  it('should handle model change', async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => mockModels
    });

    const onModelChange = jest.fn();
    render(
      <ThemeProvider>
        <ModelSelector
          selectedModel="model1"
          provider="yandex"
          onModelChange={onModelChange}
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    fireEvent.change(screen.getByRole('combobox'), {
      target: { value: 'model2' }
    });

    expect(onModelChange).toHaveBeenCalledWith('model2');
  });

  it('should show loading state', () => {
    (global.fetch as jest.Mock).mockImplementationOnce(
      () => new Promise(() => {}) // Never resolves
    );

    render(
      <ThemeProvider>
        <ModelSelector
          selectedModel="model1"
          provider="yandex"
          onModelChange={jest.fn()}
        />
      </ThemeProvider>
    );

    expect(screen.getByText('Загрузка моделей...')).toBeInTheDocument();
  });

  it('should show error state', async () => {
    (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Test error'));

    render(
      <ThemeProvider>
        <ModelSelector
          selectedModel="model1"
          provider="yandex"
          onModelChange={jest.fn()}
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('Ошибка: Test error')).toBeInTheDocument();
    });
  });

  it('should reload models when provider changes', async () => {
    const fetchMock = global.fetch as jest.Mock;
    fetchMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ['yandex1', 'yandex2']
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ['giga1', 'giga2']
      });

    const { rerender } = render(
      <ThemeProvider>
        <ModelSelector
          selectedModel="yandex1"
          provider="yandex"
          onModelChange={jest.fn()}
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('yandex1')).toBeInTheDocument();
    });

    rerender(
      <ThemeProvider>
        <ModelSelector
          selectedModel="giga1"
          provider="gigachat"
          onModelChange={jest.fn()}
        />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText('giga1')).toBeInTheDocument();
    });

    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock.mock.calls[0][0]).toContain('provider=yandex');
    expect(fetchMock.mock.calls[1][0]).toContain('provider=gigachat');
  });
}); 