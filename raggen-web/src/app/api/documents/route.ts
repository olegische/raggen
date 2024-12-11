import { NextRequest, NextResponse } from 'next/server';
import { DocumentService } from '@/services/document.service';
import { EmbedApiClient } from '@/services/embed-api';
import { BaseRepository } from '@/services/repositories/base.repository';

// Create service instances for default usage
const defaultDocumentService = new DocumentService(new BaseRepository());
const defaultEmbedApi = new EmbedApiClient();

// Helper function for date serialization
const serializeDocument = (document: any) => ({
  ...document,
  createdAt: document?.createdAt?.toISOString() || null,
  updatedAt: document?.updatedAt?.toISOString() || null
});

// For testing purposes, we allow injection of services
let documentService = defaultDocumentService;
let embedApi = defaultEmbedApi;

export const setTestServices = (mockDocService?: DocumentService, mockEmbedApi?: EmbedApiClient) => {
  documentService = mockDocService || defaultDocumentService;
  embedApi = mockEmbedApi || defaultEmbedApi;
};

export const resetServices = () => {
  documentService = defaultDocumentService;
  embedApi = defaultEmbedApi;
};

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    // Validate file type
    const fileType = file.name.split('.').pop()?.toLowerCase() || '';
    if (!documentService.isTypeSupported(fileType)) {
      const supportedTypes = documentService.getSupportedTypes();
      return NextResponse.json(
        { error: `Unsupported file type. Supported types: ${supportedTypes.join(', ')}` },
        { status: 400 }
      );
    }

    // Validate file size
    if (!documentService.isFileSizeValid(file.size)) {
      const maxSize = documentService.getMaxFileSize();
      return NextResponse.json(
        { error: `File size exceeds maximum allowed size of ${maxSize / (1024 * 1024)}MB` },
        { status: 400 }
      );
    }

    // Read file content
    const content = await file.text();

    // Process content based on file type
    const processedContent = await documentService.processContent(content, fileType);

    // Create document in database
    const document = await documentService.createDocument(
      file.name,
      fileType,
      processedContent,
      file.size,
      {
        originalName: file.name,
        mimeType: file.type,
        uploadedAt: new Date().toISOString()
      }
    );

    try {
      // Send content to raggen-embed for vectorization
      await embedApi.addDocument({
        id: document.id,
        content: processedContent
      });
    } catch (embedError) {
      // If embedding fails, delete the document and throw error
      await documentService.deleteDocument(document.id);
      throw new Error('Failed to process document for search: ' + (embedError instanceof Error ? embedError.message : String(embedError)));
    }

    return NextResponse.json(serializeDocument(document), { status: 201 });
  } catch (error) {
    console.error('Error processing file upload:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: error instanceof Error ? 400 : 500 }
    );
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (id) {
      const document = await documentService.getDocument(id);
      if (!document) {
        return NextResponse.json({ error: 'Document not found' }, { status: 404 });
      }
      return NextResponse.json(serializeDocument(document));
    }

    const documents = await documentService.getAllDocuments();
    return NextResponse.json(documents.map(serializeDocument));
  } catch (error) {
    console.error('Error getting documents:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: error instanceof Error ? 400 : 500 }
    );
  }
}

export async function DELETE(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const id = searchParams.get('id');

    if (!id) {
      return NextResponse.json({ error: 'Document ID is required' }, { status: 400 });
    }

    // Delete from database first
    const document = await documentService.deleteDocument(id);

    try {
      // Then try to remove from raggen-embed
      await embedApi.deleteDocument(id);
    } catch (embedError) {
      console.error('Failed to delete document from search index:', embedError);
      // Don't throw here since the document is already deleted from DB
    }

    return NextResponse.json(serializeDocument(document));
  } catch (error) {
    console.error('Error deleting document:', error);
    return NextResponse.json(
      { error: error instanceof Error ? error.message : 'Internal server error' },
      { status: error instanceof Error ? 400 : 500 }
    );
  }
}
