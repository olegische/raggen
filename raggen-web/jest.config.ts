import type { Config } from 'jest';
import { createJestConfig, baseConfig } from './jest.config.base';

const config: Config = {
  ...baseConfig,
  projects: [
    {
      ...baseConfig,
      displayName: 'frontend',
      testEnvironment: 'jsdom',
      testMatch: [
        '<rootDir>/src/components/**/*.test.tsx',
        '<rootDir>/src/app/**/*.test.tsx'
      ],
    },
    {
      ...baseConfig,
      displayName: 'backend',
      testEnvironment: 'node',
      testMatch: [
        '<rootDir>/src/services/**/*.test.ts',
        '<rootDir>/src/lib/**/*.test.ts',
        '<rootDir>/src/app/api/**/*.test.ts'
      ],
    },
  ],
};

export default createJestConfig(config);
