import { useState, useEffect } from 'react';

interface Document {
  id: string;
  name: string;
  type: string;
  size: number;
  createdAt: string;
  updatedAt: string;
}

interface DocumentListProps {
  onDelete?: (id: string) => Promise<void>;
  onView?: (document: Document) => void;
}

export const DocumentList = ({ onDelete, onView }: DocumentListProps) => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sortField, setSortField] = useState<keyof Document>('createdAt');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const itemsPerPage = 10;

  const fetchDocuments = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/documents');
      if (!response.ok) {
        throw new Error('Failed to fetch documents');
      }
      const data = await response.json();
      setDocuments(data);
      setTotalPages(Math.ceil(data.length / itemsPerPage));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load documents');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, []);

  const handleSort = (field: keyof Document) => {
    if (field === sortField) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const handleDelete = async (id: string) => {
    if (!onDelete) return;

    try {
      await onDelete(id);
      await fetchDocuments(); // Refresh the list
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete document');
    }
  };

  const formatFileSize = (bytes: number) => {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;

    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }

    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const sortedDocuments = [...documents].sort((a, b) => {
    if (sortField === 'size') {
      return sortDirection === 'asc' ? a.size - b.size : b.size - a.size;
    }
    
    const aValue = a[sortField];
    const bValue = b[sortField];
    
    if (sortDirection === 'asc') {
      return aValue < bValue ? -1 : aValue > bValue ? 1 : 0;
    } else {
      return aValue > bValue ? -1 : aValue < bValue ? 1 : 0;
    }
  });

  const paginatedDocuments = sortedDocuments.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const SortIcon = ({ field }: { field: keyof Document }) => (
    <span className="ml-1">
      {sortField === field ? (
        sortDirection === 'asc' ? '↑' : '↓'
      ) : (
        <span className="text-gray-300">↕</span>
      )}
    </span>
  );

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64" role="status">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-red-500 p-4 text-center" role="alert">
        {error}
      </div>
    );
  }

  if (documents.length === 0) {
    return (
      <div className="text-gray-500 p-4 text-center">
        No documents found
      </div>
    );
  }

  return (
    <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 sm:rounded-lg">
      <table className="min-w-full divide-y divide-gray-300">
        <thead className="bg-gray-50">
          <tr>
            <th
              scope="col"
              className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 cursor-pointer"
              onClick={() => handleSort('name')}
            >
              Name
              <SortIcon field="name" />
            </th>
            <th
              scope="col"
              className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 cursor-pointer"
              onClick={() => handleSort('type')}
            >
              Type
              <SortIcon field="type" />
            </th>
            <th
              scope="col"
              className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 cursor-pointer"
              onClick={() => handleSort('size')}
            >
              Size
              <SortIcon field="size" />
            </th>
            <th
              scope="col"
              className="px-3 py-3.5 text-left text-sm font-semibold text-gray-900 cursor-pointer"
              onClick={() => handleSort('createdAt')}
            >
              Created
              <SortIcon field="createdAt" />
            </th>
            <th scope="col" className="relative py-3.5 pl-3 pr-4 sm:pr-6">
              <span className="sr-only">Actions</span>
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white">
          {paginatedDocuments.map((document) => (
            <tr key={document.id}>
              <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-900">
                {document.name}
              </td>
              <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                {document.type.toUpperCase()}
              </td>
              <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                {formatFileSize(document.size)}
              </td>
              <td className="whitespace-nowrap px-3 py-4 text-sm text-gray-500">
                {formatDate(document.createdAt)}
              </td>
              <td className="relative whitespace-nowrap py-4 pl-3 pr-4 text-right text-sm font-medium sm:pr-6">
                <div className="flex justify-end gap-2">
                  {onView && (
                    <button
                      onClick={() => onView(document)}
                      className="text-blue-600 hover:text-blue-900"
                    >
                      View
                    </button>
                  )}
                  {onDelete && (
                    <button
                      onClick={() => handleDelete(document.id)}
                      className="text-red-600 hover:text-red-900 ml-4"
                    >
                      Delete
                    </button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Pagination */}
      <div className="flex items-center justify-between border-t border-gray-200 bg-white px-4 py-3 sm:px-6">
        <div className="flex flex-1 justify-between sm:hidden">
          <button
            onClick={() => setCurrentPage(page => Math.max(1, page - 1))}
            disabled={currentPage === 1}
            className="relative inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            Previous
          </button>
          <button
            onClick={() => setCurrentPage(page => Math.min(totalPages, page + 1))}
            disabled={currentPage === totalPages}
            className="relative ml-3 inline-flex items-center rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50"
          >
            Next
          </button>
        </div>
        <div className="hidden sm:flex sm:flex-1 sm:items-center sm:justify-between">
          <div>
            <p className="text-sm text-gray-700">
              Showing{' '}
              <span className="font-medium">
                {Math.min((currentPage - 1) * itemsPerPage + 1, documents.length)}
              </span>{' '}
              to{' '}
              <span className="font-medium">
                {Math.min(currentPage * itemsPerPage, documents.length)}
              </span>{' '}
              of{' '}
              <span className="font-medium">{documents.length}</span>{' '}
              results
            </p>
          </div>
          <div>
            <nav className="isolate inline-flex -space-x-px rounded-md shadow-sm" aria-label="Pagination">
              <button
                onClick={() => setCurrentPage(page => Math.max(1, page - 1))}
                disabled={currentPage === 1}
                className="relative inline-flex items-center rounded-l-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0 disabled:opacity-50"
              >
                <span className="sr-only">Previous</span>
                ←
              </button>
              {Array.from({ length: totalPages }, (_, i) => i + 1).map((page) => (
                <button
                  key={page}
                  onClick={() => setCurrentPage(page)}
                  className={`relative inline-flex items-center px-4 py-2 text-sm font-semibold ${
                    page === currentPage
                      ? 'z-10 bg-blue-600 text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600'
                      : 'text-gray-900 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0'
                  }`}
                >
                  {page}
                </button>
              ))}
              <button
                onClick={() => setCurrentPage(page => Math.min(totalPages, page + 1))}
                disabled={currentPage === totalPages}
                className="relative inline-flex items-center rounded-r-md px-2 py-2 text-gray-400 ring-1 ring-inset ring-gray-300 hover:bg-gray-50 focus:z-20 focus:outline-offset-0 disabled:opacity-50"
              >
                <span className="sr-only">Next</span>
                →
              </button>
            </nav>
          </div>
        </div>
      </div>
    </div>
  );
};
