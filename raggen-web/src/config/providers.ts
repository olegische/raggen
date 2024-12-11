import { SystemPromptType } from './prompts';

export type ProviderType = 'yandex' | 'gigachat';

export interface ProviderConfig {
  id: ProviderType;
  displayName: string;
  systemPrompt: SystemPromptType;
  apiUrl: string;
  credentials: string;
}

function getEnvironmentValue(key: string): string {
  const value = process.env[key];
  return value?.replace(/"/g, '') || '';
}

export const PROVIDER_CONFIG: Record<ProviderType, ProviderConfig> = {
  yandex: {
    id: 'yandex',
    displayName: 'YandexGPT',
    systemPrompt: 'yandex',
    apiUrl: getEnvironmentValue('YANDEX_GPT_API_URL'),
    credentials: getEnvironmentValue('YANDEX_API_KEY')
  },
  gigachat: {
    id: 'gigachat',
    displayName: 'GigaChat',
    systemPrompt: 'gigachat',
    apiUrl: getEnvironmentValue('GIGACHAT_API_URL'),
    credentials: getEnvironmentValue('GIGACHAT_CREDENTIALS')
  }
} as const;

export function getProviderConfig(providerId: ProviderType): ProviderConfig {
  const config = PROVIDER_CONFIG[providerId];
  if (!config) {
    throw new Error(`Unknown provider: ${providerId}`);
  }
  
  if (providerId === 'gigachat' && (!config.apiUrl || !config.credentials)) {
    throw new Error('Missing GigaChat configuration');
  }

  return config;
}

export function getProviderDisplayName(providerId: ProviderType): string {
  if (!providerId || !PROVIDER_CONFIG[providerId]) {
    console.warn(`Unknown provider: ${providerId}, falling back to default name`);
    return String(providerId || 'Unknown');
  }
  return PROVIDER_CONFIG[providerId].displayName;
}
