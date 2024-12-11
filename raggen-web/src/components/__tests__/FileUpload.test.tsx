import { render, screen, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { FileUpload } from '../FileUpload';
import { DocumentService } from '../../services/document.service';
import '@testing-library/jest-dom';

// Create a mock DocumentService class with Jest functions
const createMockDocumentService = () => ({
  isTypeSupported: jest.fn((type: string) => {
    console.log('isTypeSupported called with:', type);
    return ['txt', 'md', 'html'].includes(type);
  }),
  isFileSizeValid: jest.fn((size: number) => size <= 10 * 1024 * 1024),
  getSupportedTypes: jest.fn(() => ['txt', 'md', 'html']),
  getMaxFileSize: jest.fn(() => 10 * 1024 * 1024)
});

describe('FileUpload', () => {
  const mockOnFileSelect = jest.fn();
  const mockOnError = jest.fn();
  let mockDocumentService: ReturnType<typeof createMockDocumentService>;

  beforeEach(() => {
    jest.clearAllMocks();
    mockDocumentService = createMockDocumentService();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it('renders upload area with instructions', () => {
    render(<FileUpload onFileSelect={mockOnFileSelect} />);
    
    expect(screen.getByText(/Click to upload/i)).toBeInTheDocument();
    expect(screen.getByText(/drag and drop/i)).toBeInTheDocument();
    expect(screen.getByText(/Supported files:/i)).toBeInTheDocument();
  });

  it('handles file input change', async () => {
    render(
      <FileUpload 
        onFileSelect={mockOnFileSelect} 
        onError={mockOnError}
        documentService={mockDocumentService as unknown as DocumentService}
      />
    );

    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const input = screen.getByRole('button').querySelector('input[type="file"]') as HTMLInputElement;

    await userEvent.upload(input, file);

    expect(mockOnFileSelect).toHaveBeenCalledWith(file);
    expect(mockOnError).not.toHaveBeenCalled();
  });

  it('validates file type', async () => {
    // Force the mock to return false for any input
    mockDocumentService.isTypeSupported.mockImplementation((type: string) => {
      console.log('Mock isTypeSupported called with:', type);
      return false;
    });

    render(
      <FileUpload 
        onFileSelect={mockOnFileSelect} 
        onError={mockOnError}
        documentService={mockDocumentService as unknown as DocumentService}
      />
    );

    const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });
    const input = screen.getByRole('button').querySelector('input[type="file"]') as HTMLInputElement;

    await userEvent.upload(input, file);

    // Add debug logs
    console.log('mockOnError calls:', mockOnError.mock.calls);
    console.log('isTypeSupported calls:', mockDocumentService.isTypeSupported.mock.calls);

    expect(mockOnFileSelect).not.toHaveBeenCalled();
    expect(mockOnError).toHaveBeenCalledWith(expect.stringContaining('Unsupported file type'));

    // Wait for the error message to appear
    const errorMessage = await screen.findByTestId('error-message');
    expect(errorMessage).toHaveTextContent(/Unsupported file type/i);
  });

  it('validates file size', async () => {
    mockDocumentService.isFileSizeValid.mockReturnValue(false);

    render(
      <FileUpload 
        onFileSelect={mockOnFileSelect} 
        onError={mockOnError}
        documentService={mockDocumentService as unknown as DocumentService}
      />
    );

    const largeContent = 'a'.repeat(11 * 1024 * 1024); // 11MB
    const file = new File([largeContent], 'large.txt', { type: 'text/plain' });
    const input = screen.getByRole('button').querySelector('input[type="file"]') as HTMLInputElement;

    await userEvent.upload(input, file);

    expect(mockOnFileSelect).not.toHaveBeenCalled();
    expect(mockOnError).toHaveBeenCalledWith(expect.stringContaining('File size exceeds'));

    // Wait for the error message to appear
    const errorMessage = await screen.findByTestId('error-message');
    expect(errorMessage).toHaveTextContent(/File size exceeds/i);
  });

  it('handles drag and drop', () => {
    render(
      <FileUpload 
        onFileSelect={mockOnFileSelect}
        documentService={mockDocumentService as unknown as DocumentService}
      />
    );
    const dropzone = screen.getByRole('button');

    // Test drag enter
    fireEvent.dragEnter(dropzone);
    expect(dropzone).toHaveClass('border-blue-500');
    expect(dropzone).toHaveClass('bg-blue-50');

    // Test drag leave
    fireEvent.dragLeave(dropzone);
    expect(dropzone).not.toHaveClass('border-blue-500');
    expect(dropzone).not.toHaveClass('bg-blue-50');

    // Test file drop
    const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
    const dataTransfer = {
      files: [file],
      items: [
        {
          kind: 'file',
          type: file.type,
          getAsFile: () => file
        }
      ],
      types: ['Files']
    };

    fireEvent.drop(dropzone, { dataTransfer });
    expect(mockOnFileSelect).toHaveBeenCalledWith(file);
  });

  it('handles multiple files by taking only the first one', async () => {
    render(
      <FileUpload 
        onFileSelect={mockOnFileSelect}
        documentService={mockDocumentService as unknown as DocumentService}
      />
    );

    const file1 = new File(['content 1'], 'test1.txt', { type: 'text/plain' });
    const file2 = new File(['content 2'], 'test2.txt', { type: 'text/plain' });
    const input = screen.getByRole('button').querySelector('input[type="file"]') as HTMLInputElement;

    await userEvent.upload(input, [file1, file2]);

    expect(mockOnFileSelect).toHaveBeenCalledTimes(1);
    expect(mockOnFileSelect).toHaveBeenCalledWith(file1);
  });

  it('clears error when valid file is selected after invalid one', async () => {
    // Set up mock to return false for first call and true for second call
    mockDocumentService.isTypeSupported
      .mockImplementationOnce(() => false)
      .mockImplementationOnce(() => true);

    render(
      <FileUpload 
        onFileSelect={mockOnFileSelect} 
        onError={mockOnError}
        documentService={mockDocumentService as unknown as DocumentService}
      />
    );
    
    const input = screen.getByRole('button').querySelector('input[type="file"]') as HTMLInputElement;

    // First upload invalid file
    const invalidFile = new File(['test'], 'test.pdf', { type: 'application/pdf' });
    await userEvent.upload(input, invalidFile);

    // Wait for the error message to appear
    const errorMessage = await screen.findByTestId('error-message');
    expect(errorMessage).toHaveTextContent(/Unsupported file type/i);

    // Then upload valid file
    const validFile = new File(['test'], 'test.txt', { type: 'text/plain' });
    await userEvent.upload(input, validFile);
    expect(screen.queryByTestId('error-message')).not.toBeInTheDocument();
  });
});
