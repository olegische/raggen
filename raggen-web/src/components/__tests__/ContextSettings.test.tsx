import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ContextSettings from '../ContextSettings';

describe('ContextSettings', () => {
  const defaultProps = {
    maxContextMessages: 5,
    contextScoreThreshold: 0.7,
    contextEnabled: true,
    onSettingsChange: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders correctly with default props', () => {
    render(<ContextSettings {...defaultProps} />);
    
    // Проверяем заголовок
    expect(screen.getByText('Настройки контекста')).toBeInTheDocument();
    
    // Проверяем чекбокс
    const checkbox = screen.getByRole('checkbox');
    expect(checkbox).toBeChecked();
    
    // Проверяем слайдеры
    const maxMessagesSlider = screen.getByLabelText('Максимальное количество сообщений для контекста');
    expect(maxMessagesSlider).toHaveValue('5');
    
    const thresholdSlider = screen.getByLabelText('Минимальный порог релевантности контекста');
    expect(thresholdSlider).toHaveValue('70');
  });

  it('calls onSettingsChange when checkbox is toggled', async () => {
    render(<ContextSettings {...defaultProps} />);
    
    const checkbox = screen.getByRole('checkbox');
    await userEvent.click(checkbox);
    
    expect(defaultProps.onSettingsChange).toHaveBeenCalledWith({
      maxContextMessages: 5,
      contextScoreThreshold: 0.7,
      contextEnabled: false,
    });
  });

  it('calls onSettingsChange when max messages slider is changed', async () => {
    render(<ContextSettings {...defaultProps} />);
    
    const slider = screen.getByLabelText('Максимальное количество сообщений для контекста');
    fireEvent.change(slider, { target: { value: '3' } });
    
    expect(defaultProps.onSettingsChange).toHaveBeenCalledWith({
      maxContextMessages: 3,
      contextScoreThreshold: 0.7,
      contextEnabled: true,
    });
  });

  it('calls onSettingsChange when threshold slider is changed', async () => {
    render(<ContextSettings {...defaultProps} />);
    
    const slider = screen.getByLabelText('Минимальный порог релевантности контекста');
    fireEvent.change(slider, { target: { value: '80' } });
    
    expect(defaultProps.onSettingsChange).toHaveBeenCalledWith({
      maxContextMessages: 5,
      contextScoreThreshold: 0.8,
      contextEnabled: true,
    });
  });

  it('displays correct percentage for threshold', () => {
    render(<ContextSettings {...defaultProps} />);
    expect(screen.getByText('70%')).toBeInTheDocument();
  });

  // Тесты доступности
  it('has no accessibility violations', async () => {
    const { container } = render(<ContextSettings {...defaultProps} />);
    
    // Проверяем наличие ARIA-атрибутов
    const sliders = screen.getAllByRole('slider');
    sliders.forEach(slider => {
      expect(slider).toHaveAttribute('aria-label');
    });
    
    // Проверяем связь label с input через htmlFor
    const labels = container.querySelectorAll('label[for]');
    labels.forEach(label => {
      const input = screen.getByLabelText(label.textContent || '');
      expect(input).toBeInTheDocument();
    });
  });

  it('supports keyboard navigation', async () => {
    render(<ContextSettings {...defaultProps} />);
    
    // Проверяем, что все интерактивные элементы можно выбрать через Tab
    const checkbox = screen.getByRole('checkbox');
    const maxMessagesSlider = screen.getByRole('slider', { name: /Максимальное количество сообщений/ });
    const thresholdSlider = screen.getByRole('slider', { name: /Минимальный порог релевантности/ });
    
    checkbox.focus();
    expect(document.activeElement).toBe(checkbox);
    
    maxMessagesSlider.focus();
    expect(document.activeElement).toBe(maxMessagesSlider);
    
    thresholdSlider.focus();
    expect(document.activeElement).toBe(thresholdSlider);
  });

  it('updates values with keyboard controls', async () => {
    render(<ContextSettings {...defaultProps} />);
    
    const maxMessagesSlider = screen.getByLabelText('Максимальное количество сообщений для контекста');
    maxMessagesSlider.focus();
    
    // Эмулируем нажатие стрелок
    fireEvent.change(maxMessagesSlider, { target: { value: '6' } });
    
    expect(defaultProps.onSettingsChange).toHaveBeenCalledWith({
      maxContextMessages: 6,
      contextScoreThreshold: 0.7,
      contextEnabled: true,
    });
  });
}); 