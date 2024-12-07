import { render, screen } from '@testing-library/react';
import ContextIndicator from '../ContextIndicator';
import { ContextSearchResult } from '@/services/context.service';

describe('ContextIndicator', () => {
  const mockContext: ContextSearchResult[] = [
    {
      message: {
        id: 1,
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

  it('renders context information correctly', () => {
    render(<ContextIndicator context={mockContext} />);
    
    expect(screen.getByText('Использованный контекст:')).toBeInTheDocument();
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('Previous relevant message')).toBeInTheDocument();
  });

  it('does not render when context is empty', () => {
    const { container } = render(<ContextIndicator context={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('does not render when context is undefined', () => {
    // @ts-expect-error - testing undefined case explicitly
    const { container } = render(<ContextIndicator context={undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('displays multiple context items correctly', () => {
    const multipleContext: ContextSearchResult[] = [
      ...mockContext,
      {
        message: {
          id: 2,
          chatId: 'chat-1',
          message: 'Another relevant message',
          model: 'test-model',
          provider: 'test-provider',
          temperature: 0.7,
          maxTokens: 1000,
          timestamp: new Date(),
          response: null
        },
        score: 0.75,
        usedInPrompt: true
      }
    ];

    render(<ContextIndicator context={multipleContext} />);
    
    expect(screen.getByText('85%')).toBeInTheDocument();
    expect(screen.getByText('75%')).toBeInTheDocument();
    expect(screen.getByText('Previous relevant message')).toBeInTheDocument();
    expect(screen.getByText('Another relevant message')).toBeInTheDocument();
  });
}); 