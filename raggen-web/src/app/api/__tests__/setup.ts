import { PrismaClient } from '@prisma/client'
import { beforeAll, afterAll } from '@jest/globals'

const prisma = new PrismaClient()

beforeAll(async () => {
  // Connect to the database before all tests
  await prisma.$connect()
})

afterAll(async () => {
  // Disconnect from the database after all tests
  await prisma.$disconnect()
})

export { prisma }
