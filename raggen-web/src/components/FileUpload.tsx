import { useState, useCallback, useRef } from 'react';
import { DocumentService } from '../services/document.service';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  onError?: (error: string) => void;
  className?: string;
  documentService?: DocumentService; // Make service injectable for testing
}

export const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  onError,
  className = '',
  documentService = new DocumentService() // Default to new instance if not provided
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const validateFile = useCallback((file: File): boolean => {
    // Get file extension
    const extension = file.name.split('.').pop()?.toLowerCase() || '';
    
    // Check file type
    if (!documentService.isTypeSupported(extension)) {
      const supportedTypes = documentService.getSupportedTypes();
      const errorMessage = `Unsupported file type. Supported types: ${supportedTypes.join(', ')}`;
      setError(errorMessage);
      onError?.(errorMessage);
      return false;
    }

    // Check file size
    if (!documentService.isFileSizeValid(file.size)) {
      const maxSize = documentService.getMaxFileSize();
      const errorMessage = `File size exceeds maximum allowed size of ${maxSize / (1024 * 1024)}MB`;
      setError(errorMessage);
      onError?.(errorMessage);
      return false;
    }

    setError(null);
    return true;
  }, [documentService, onError]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;

    const file = files[0]; // Take only the first file
    if (validateFile(file)) {
      onFileSelect(file);
    }
  }, [onFileSelect, validateFile]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const file = files[0]; // Take only the first file
    if (validateFile(file)) {
      onFileSelect(file);
    }
  }, [onFileSelect, validateFile]);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className={className}>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload file"
        className={`relative border-2 border-dashed rounded-lg p-8 text-center cursor-pointer
          ${isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'}
          transition-colors duration-200 ease-in-out`}
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
        onClick={handleClick}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            handleClick();
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={handleFileInput}
          accept={documentService.getSupportedTypes().map(type => `.${type}`).join(',')}
          aria-hidden="true"
        />
        
        <div className="space-y-2">
          <svg
            className={`mx-auto h-12 w-12 ${isDragging ? 'text-blue-500' : 'text-gray-400'}`}
            stroke="currentColor"
            fill="none"
            viewBox="0 0 48 48"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M24 14v20m-10-10h20"
            />
          </svg>
          
          <div className="text-sm text-gray-600">
            <span className="font-medium">Click to upload</span> or drag and drop
          </div>
          
          <div className="text-xs text-gray-500">
            Supported files: {documentService.getSupportedTypes().join(', ')}
          </div>
        </div>
      </div>

      {error && (
        <div className="mt-2 text-sm text-red-600" role="alert" data-testid="error-message">
          {error}
        </div>
      )}
    </div>
  );
};
