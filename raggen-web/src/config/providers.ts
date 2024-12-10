import { SystemPromptType } from './prompts';

export type ProviderType = 'yandex' | 'gigachat';

export interface ProviderConfig {
  id: ProviderType;
  displayName: string;
  systemPrompt: SystemPromptType;
}

export const PROVIDER_CONFIG: Record<ProviderType, ProviderConfig> = {
  yandex: {
    id: 'yandex',
    displayName: 'YandexGPT',
    systemPrompt: 'yandex'
  },
  gigachat: {
    id: 'gigachat',
    displayName: 'GigaChat',
    systemPrompt: 'gigachat'
  }
} as const;

export function getProviderDisplayName(providerId: ProviderType): string {
  if (!providerId || !PROVIDER_CONFIG[providerId]) {
    console.warn(`Unknown provider: ${providerId}, falling back to default name`);
    return String(providerId || 'Unknown');
  }
  return PROVIDER_CONFIG[providerId].displayName;
}

export function getProviderConfig(providerId: ProviderType): ProviderConfig {
  const config = PROVIDER_CONFIG[providerId];
  if (!config) {
    throw new Error(`Unknown provider: ${providerId}`);
  }
  return config;
}
