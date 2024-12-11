import { PrismaClient } from '@prisma/client';
import prismaClient from '../../lib/db';

export class BaseRepository {
  protected readonly prisma: PrismaClient;

  constructor(client?: PrismaClient) {
    this.prisma = client || prismaClient;
  }

  createRepository<T extends BaseRepository>(
    RepositoryClass: new (prisma: PrismaClient) => T
  ): T {
    return new RepositoryClass(this.prisma);
  }
}
