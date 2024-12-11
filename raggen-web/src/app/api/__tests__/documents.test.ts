import { beforeEach, describe, expect, it } from '@jest/globals'
import { prisma } from './setup'

describe('Document and Chunk Models', () => {
  beforeEach(async () => {
    // Clean up the test database before each test
    await prisma.chunk.deleteMany()
    await prisma.document.deleteMany()
  })

  describe('Document CRUD', () => {
    it('should create a document', async () => {
      const document = await prisma.document.create({
        data: {
          name: 'test.txt',
          type: 'txt',
          size: 100,
          content: 'Test content',
          metadata: JSON.stringify({ author: 'Test Author' })
        }
      })

      expect(document).toBeDefined()
      expect(document.name).toBe('test.txt')
      expect(document.type).toBe('txt')
      expect(document.content).toBe('Test content')
      expect(JSON.parse(document.metadata!)).toEqual({ author: 'Test Author' })
    })

    it('should read a document', async () => {
      // Create a document
      const created = await prisma.document.create({
        data: {
          name: 'test.txt',
          type: 'txt',
          size: 100,
          content: 'Test content'
        }
      })

      // Read the document
      const document = await prisma.document.findUnique({
        where: { id: created.id }
      })

      expect(document).toBeDefined()
      expect(document?.name).toBe('test.txt')
    })

    it('should update a document', async () => {
      // Create a document
      const created = await prisma.document.create({
        data: {
          name: 'test.txt',
          type: 'txt',
          size: 100,
          content: 'Test content'
        }
      })

      // Update the document
      const updated = await prisma.document.update({
        where: { id: created.id },
        data: { name: 'updated.txt' }
      })

      expect(updated.name).toBe('updated.txt')
    })

    it('should delete a document', async () => {
      // Create a document
      const created = await prisma.document.create({
        data: {
          name: 'test.txt',
          type: 'txt',
          size: 100,
          content: 'Test content'
        }
      })

      // Delete the document
      await prisma.document.delete({
        where: { id: created.id }
      })

      // Try to find the deleted document
      const document = await prisma.document.findUnique({
        where: { id: created.id }
      })

      expect(document).toBeNull()
    })
  })

  describe('Document-Chunk Relationship', () => {
    it('should create a document with chunks', async () => {
      // Create a document with chunks
      const document = await prisma.document.create({
        data: {
          name: 'test.txt',
          type: 'txt',
          size: 100,
          content: 'Test content',
          chunks: {
            create: [
              {
                content: 'Chunk 1',
                metadata: JSON.stringify({ position: 0 })
              },
              {
                content: 'Chunk 2',
                metadata: JSON.stringify({ position: 1 })
              }
            ]
          }
        },
        include: {
          chunks: true
        }
      })

      expect(document.chunks).toHaveLength(2)
      expect(document.chunks[0].content).toBe('Chunk 1')
      expect(JSON.parse(document.chunks[0].metadata!)).toEqual({ position: 0 })
    })

    it('should cascade delete chunks when document is deleted', async () => {
      // Create a document with chunks
      const document = await prisma.document.create({
        data: {
          name: 'test.txt',
          type: 'txt',
          size: 100,
          content: 'Test content',
          chunks: {
            create: [
              { content: 'Chunk 1' },
              { content: 'Chunk 2' }
            ]
          }
        }
      })

      // Delete the document
      await prisma.document.delete({
        where: { id: document.id }
      })

      // Check if chunks were deleted
      const chunks = await prisma.chunk.findMany({
        where: { documentId: document.id }
      })

      expect(chunks).toHaveLength(0)
    })
  })
})
