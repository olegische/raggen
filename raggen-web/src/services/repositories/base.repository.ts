import { PrismaClient } from '@prisma/client';

export class BaseRepository {
  protected readonly prisma: PrismaClient;

  constructor(prismaClient?: PrismaClient) {
    this.prisma = prismaClient || new PrismaClient();
  }

  createRepository<T extends BaseRepository>(
    RepositoryClass: new (prisma: PrismaClient) => T
  ): T {
    return new RepositoryClass(this.prisma);
  }
}
