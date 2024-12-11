import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { DocumentList } from '../DocumentList';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

const mockDate = new Date('2024-01-01T10:00:00.000Z');

const createMockDocument = (id: string) => ({
  id,
  name: `document${id}.txt`,
  type: 'txt',
  size: 1024,
  createdAt: mockDate.toISOString(),
  updatedAt: mockDate.toISOString()
});

describe('DocumentList', () => {
  beforeEach(() => {
    mockFetch.mockClear();
  });

  it('renders loading state initially', () => {
    mockFetch.mockImplementationOnce(() => new Promise(() => {}));
    render(<DocumentList />);
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('renders error state when fetch fails', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Failed to fetch'));
    render(<DocumentList />);
    await waitFor(() => {
      expect(screen.getByRole('alert')).toHaveTextContent('Failed to fetch');
    });
  });

  it('renders empty state when no documents', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => []
    });

    render(<DocumentList />);
    await waitFor(() => {
      expect(screen.getByText('No documents found')).toBeInTheDocument();
    });
  });

  it('renders documents list', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [createMockDocument('1'), createMockDocument('2')]
    });

    render(<DocumentList />);

    await waitFor(() => {
      expect(screen.getByText('document1.txt')).toBeInTheDocument();
      expect(screen.getByText('document2.txt')).toBeInTheDocument();
    });
  });

  it('handles sorting by different fields', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [createMockDocument('1'), createMockDocument('2')]
    });

    render(<DocumentList />);

    // Wait for documents to load
    await waitFor(() => {
      expect(screen.getByText('document1.txt')).toBeInTheDocument();
    });

    // Click name header to sort
    const nameHeader = screen.getByText('Name').closest('th');
    fireEvent.click(nameHeader!);

    // Verify sort indicators
    expect(nameHeader).toHaveTextContent('↑');

    // Click again to reverse sort
    fireEvent.click(nameHeader!);
    expect(nameHeader).toHaveTextContent('↓');
  });

  it('handles delete action', async () => {
    const onDelete = jest.fn().mockResolvedValue(undefined);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [createMockDocument('1'), createMockDocument('2')]
    }).mockResolvedValueOnce({
      ok: true,
      json: async () => [createMockDocument('2')]
    });

    render(<DocumentList onDelete={onDelete} />);

    await waitFor(() => {
      expect(screen.getAllByText('Delete')[0]).toBeInTheDocument();
    });

    // Click delete button
    fireEvent.click(screen.getAllByText('Delete')[0]);

    expect(onDelete).toHaveBeenCalledWith('1');

    // Wait for the list to refresh
    await waitFor(() => {
      expect(screen.queryByText('document1.txt')).not.toBeInTheDocument();
      expect(screen.getByText('document2.txt')).toBeInTheDocument();
    });
  });

  it('handles view action', async () => {
    const onView = jest.fn();
    const mockDoc = createMockDocument('1');
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => [mockDoc]
    });

    render(<DocumentList onView={onView} />);

    await waitFor(() => {
      expect(screen.getAllByText('View')[0]).toBeInTheDocument();
    });

    // Click view button
    fireEvent.click(screen.getAllByText('View')[0]);

    expect(onView).toHaveBeenCalledWith(mockDoc);
  });

  it('handles pagination', async () => {
    // Create 15 mock documents to test pagination
    const manyDocuments = Array.from({ length: 15 }, (_, i) => ({
      ...createMockDocument(String(i + 1)),
      name: `document${i + 1}.txt`
    }));

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => manyDocuments
    });

    render(<DocumentList />);

    // Wait for documents to load
    await waitFor(() => {
      expect(screen.getByText('document1.txt')).toBeInTheDocument();
    });

    // Verify first page shows first 10 documents
    expect(screen.getByText('document1.txt')).toBeInTheDocument();
    expect(screen.getByText('document10.txt')).toBeInTheDocument();
    expect(screen.queryByText('document11.txt')).not.toBeInTheDocument();

    // Go to next page
    fireEvent.click(screen.getByText('2'));

    // Verify second page shows remaining documents
    expect(screen.getByText('document11.txt')).toBeInTheDocument();
    expect(screen.getByText('document15.txt')).toBeInTheDocument();
    expect(screen.queryByText('document1.txt')).not.toBeInTheDocument();
  });

  it('formats file size correctly', async () => {
    const documentsWithDifferentSizes = [
      { ...createMockDocument('1'), size: 500 }, // 500 B
      { ...createMockDocument('2'), size: 1024 * 1024 }, // 1 MB
    ];

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => documentsWithDifferentSizes
    });

    render(<DocumentList />);

    await waitFor(() => {
      expect(screen.getByText('500.0 B')).toBeInTheDocument();
      expect(screen.getByText('1.0 MB')).toBeInTheDocument();
    });
  });

  it('refreshes list after delete', async () => {
    const onDelete = jest.fn().mockResolvedValue(undefined);
    
    // First fetch returns both documents
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [createMockDocument('1'), createMockDocument('2')]
      })
      // Second fetch (after delete) returns only one document
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [createMockDocument('2')]
      });

    render(<DocumentList onDelete={onDelete} />);

    await waitFor(() => {
      expect(screen.getByText('document1.txt')).toBeInTheDocument();
    });

    // Delete first document
    fireEvent.click(screen.getAllByText('Delete')[0]);

    await waitFor(() => {
      expect(screen.queryByText('document1.txt')).not.toBeInTheDocument();
      expect(screen.getByText('document2.txt')).toBeInTheDocument();
    });
  });
});
