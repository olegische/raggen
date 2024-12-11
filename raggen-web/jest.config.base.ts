import type { Config } from 'jest';
import nextJest from 'next/jest';

const createJestConfig = nextJest({
  dir: './',
});

const baseConfig: Config = {
  setupFilesAfterEnv: ['<rootDir>/jest.setup.js'],
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
  },
  transform: {
    '^.+\\.(js|jsx|ts|tsx|mjs)$': ['babel-jest', {
      presets: ['next/babel']
    }]
  },
  transformIgnorePatterns: [
    '/node_modules/(?!next-themes|@testing-library|date-fns)/'
  ],
};

export { createJestConfig, baseConfig };
