import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider } from 'next-themes';
import Footer from '../Footer';

describe('Footer', () => {
  const defaultProps = {
    provider: 'yandex' as const,
    temperature: 0.7,
    maxTokens: 1000,
    onSettingsChange: jest.fn()
  };

  it('should render footer with provider name', () => {
    render(
      <ThemeProvider>
        <Footer {...defaultProps} />
      </ThemeProvider>
    );

    expect(screen.getByText(/Настройки yandex/i)).toBeInTheDocument();
  });

  it('should toggle settings panel', () => {
    render(
      <ThemeProvider>
        <Footer {...defaultProps} />
      </ThemeProvider>
    );

    const button = screen.getByRole('button');
    fireEvent.click(button);

    expect(screen.getByText(/Температура:/i)).toBeInTheDocument();
    expect(screen.getByText(/Максимум токенов:/i)).toBeInTheDocument();
  });

  it('should call onSettingsChange when temperature changes', () => {
    render(
      <ThemeProvider>
        <Footer {...defaultProps} />
      </ThemeProvider>
    );

    fireEvent.click(screen.getByRole('button'));
    
    const temperatureInput = screen.getByLabelText(/Температура:/i);
    fireEvent.change(temperatureInput, { target: { value: '0.8' } });

    expect(defaultProps.onSettingsChange).toHaveBeenCalledWith({
      temperature: 0.8,
      maxTokens: 1000
    });
  });

  it('should call onSettingsChange when maxTokens changes', () => {
    render(
      <ThemeProvider>
        <Footer {...defaultProps} />
      </ThemeProvider>
    );

    fireEvent.click(screen.getByRole('button'));
    
    const maxTokensInput = screen.getByLabelText(/Максимум токенов:/i);
    fireEvent.change(maxTokensInput, { target: { value: '1500' } });

    expect(defaultProps.onSettingsChange).toHaveBeenCalledWith({
      temperature: 0.7,
      maxTokens: 1500
    });
  });

  it('should disable controls when disabled prop is true', () => {
    render(
      <ThemeProvider>
        <Footer {...defaultProps} disabled />
      </ThemeProvider>
    );

    expect(screen.getByRole('button')).toBeDisabled();
  });
}); 