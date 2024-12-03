export const GENERATION_CONFIG = {
  temperature: {
    default: 0.3,
    min: 0,
    max: 1,
    step: 0.1
  },
  maxTokens: {
    default: 2000,
    min: 100,
    max: 4000,
    step: 100
  }
} as const;

export type GenerationConfig = typeof GENERATION_CONFIG; 