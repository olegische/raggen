import { NextRequest } from 'next/server';
import { POST, GET, DELETE, setTestServices, resetServices } from '../documents/route';
import { DocumentService } from '@/services/document.service';
import { EmbedApiClient } from '@/services/embed-api';

// Mock the services
jest.mock('@/services/document.service');
jest.mock('@/services/embed-api');
jest.mock('@/services/repositories/base.repository');

const mockDate = new Date('2024-01-01T00:00:00.000Z');

const createMockDocument = (id: string) => ({
  id,
  name: `doc${id}.txt`,
  type: 'txt',
  size: 100,
  content: 'test content',
  metadata: JSON.stringify({ originalName: `doc${id}.txt` }),
  createdAt: mockDate,
  updatedAt: mockDate
});

describe('Documents API', () => {
  let mockDocumentService: jest.Mocked<DocumentService>;
  let mockEmbedApi: jest.Mocked<EmbedApiClient>;

  beforeEach(() => {
    // Clear all mocks
    jest.clearAllMocks();

    // Setup DocumentService mock
    mockDocumentService = {
      isTypeSupported: jest.fn(),
      isFileSizeValid: jest.fn(),
      processContent: jest.fn(),
      createDocument: jest.fn(),
      getDocument: jest.fn(),
      getAllDocuments: jest.fn(),
      deleteDocument: jest.fn(),
      getSupportedTypes: jest.fn(),
      getMaxFileSize: jest.fn(),
    } as unknown as jest.Mocked<DocumentService>;

    // Setup EmbedApiClient mock
    mockEmbedApi = {
      addDocument: jest.fn(),
      deleteDocument: jest.fn(),
    } as unknown as jest.Mocked<EmbedApiClient>;

    // Setup default mock implementations
    mockDocumentService.getSupportedTypes.mockReturnValue(['txt', 'md', 'html']);
    mockDocumentService.getMaxFileSize.mockReturnValue(10 * 1024 * 1024);

    // Inject mocked services
    setTestServices(mockDocumentService, mockEmbedApi);
  });

  afterEach(() => {
    // Reset services to defaults
    resetServices();
  });

  describe('POST /api/documents', () => {
    it('should handle file upload successfully', async () => {
      // Setup mocks
      mockDocumentService.isTypeSupported.mockReturnValue(true);
      mockDocumentService.isFileSizeValid.mockReturnValue(true);
      mockDocumentService.processContent.mockResolvedValue('processed content');
      mockDocumentService.createDocument.mockResolvedValue(createMockDocument('123'));
      mockEmbedApi.addDocument.mockResolvedValue();

      // Create test file
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const formData = new FormData();
      formData.append('file', file);

      // Create request
      const request = new NextRequest('http://localhost/api/documents', {
        method: 'POST',
        body: formData,
      });

      // Make request
      const response = await POST(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(201);
      expect(data.name).toBe('doc123.txt');
      expect(data.createdAt).toBe(mockDate.toISOString());
      expect(data.updatedAt).toBe(mockDate.toISOString());
      expect(mockDocumentService.createDocument).toHaveBeenCalled();
      expect(mockEmbedApi.addDocument).toHaveBeenCalled();
    });

    it('should handle invalid file type', async () => {
      // Setup mocks
      mockDocumentService.isTypeSupported.mockReturnValue(false);

      // Create test file
      const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });
      const formData = new FormData();
      formData.append('file', file);

      // Create request
      const request = new NextRequest('http://localhost/api/documents', {
        method: 'POST',
        body: formData,
      });

      // Make request
      const response = await POST(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(400);
      expect(data.error).toContain('Unsupported file type');
      expect(mockDocumentService.createDocument).not.toHaveBeenCalled();
      expect(mockEmbedApi.addDocument).not.toHaveBeenCalled();
    });

    it('should handle file size exceeding limit', async () => {
      // Setup mocks
      mockDocumentService.isTypeSupported.mockReturnValue(true);
      mockDocumentService.isFileSizeValid.mockReturnValue(false);

      // Create test file
      const file = new File(['test content'], 'test.txt', { type: 'text/plain' });
      const formData = new FormData();
      formData.append('file', file);

      // Create request
      const request = new NextRequest('http://localhost/api/documents', {
        method: 'POST',
        body: formData,
      });

      // Make request
      const response = await POST(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(400);
      expect(data.error).toContain('File size exceeds');
      expect(mockDocumentService.createDocument).not.toHaveBeenCalled();
      expect(mockEmbedApi.addDocument).not.toHaveBeenCalled();
    });
  });

  describe('GET /api/documents', () => {
    it('should get all documents', async () => {
      // Setup mock
      const mockDocuments = [
        createMockDocument('1'),
        createMockDocument('2')
      ];
      mockDocumentService.getAllDocuments.mockResolvedValue(mockDocuments);

      // Create request
      const request = new NextRequest('http://localhost/api/documents');

      // Make request
      const response = await GET(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(200);
      expect(data).toEqual(mockDocuments.map(doc => ({
        ...doc,
        createdAt: mockDate.toISOString(),
        updatedAt: mockDate.toISOString()
      })));
    });

    it('should get document by id', async () => {
      // Setup mock
      const mockDocument = createMockDocument('1');
      mockDocumentService.getDocument.mockResolvedValue(mockDocument);

      // Create request
      const request = new NextRequest('http://localhost/api/documents?id=1');

      // Make request
      const response = await GET(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(200);
      expect(data).toEqual({
        ...mockDocument,
        createdAt: mockDate.toISOString(),
        updatedAt: mockDate.toISOString()
      });
    });

    it('should handle document not found', async () => {
      // Setup mock
      mockDocumentService.getDocument.mockResolvedValue(null);

      // Create request
      const request = new NextRequest('http://localhost/api/documents?id=999');

      // Make request
      const response = await GET(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(404);
      expect(data.error).toBe('Document not found');
    });
  });

  describe('DELETE /api/documents', () => {
    it('should delete document successfully', async () => {
      // Setup mock
      const mockDocument = createMockDocument('1');
      mockDocumentService.deleteDocument.mockResolvedValue(mockDocument);
      mockEmbedApi.deleteDocument.mockResolvedValue();

      // Create request
      const request = new NextRequest('http://localhost/api/documents?id=1');

      // Make request
      const response = await DELETE(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(200);
      expect(data).toEqual({
        ...mockDocument,
        createdAt: mockDate.toISOString(),
        updatedAt: mockDate.toISOString()
      });
      expect(mockDocumentService.deleteDocument).toHaveBeenCalledWith('1');
      expect(mockEmbedApi.deleteDocument).toHaveBeenCalledWith('1');
    });

    it('should handle missing document id', async () => {
      // Create request without id
      const request = new NextRequest('http://localhost/api/documents');

      // Make request
      const response = await DELETE(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(400);
      expect(data.error).toBe('Document ID is required');
      expect(mockDocumentService.deleteDocument).not.toHaveBeenCalled();
      expect(mockEmbedApi.deleteDocument).not.toHaveBeenCalled();
    });

    it('should handle document deletion error', async () => {
      // Setup mock to throw error
      mockDocumentService.deleteDocument.mockRejectedValue(new Error('Failed to delete'));

      // Create request
      const request = new NextRequest('http://localhost/api/documents?id=1');

      // Make request
      const response = await DELETE(request);
      const data = await response.json();

      // Verify response
      expect(response.status).toBe(400);
      expect(data.error).toBe('Failed to delete');
    });
  });
});
