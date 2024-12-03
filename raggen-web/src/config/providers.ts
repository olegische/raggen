import { ProviderType } from '@/providers/factory';

export const PROVIDER_CONFIG = {
  yandex: {
    id: 'yandex' as ProviderType,
    displayName: 'Yandex'
  },
  gigachat: {
    id: 'gigachat' as ProviderType,
    displayName: 'GigaChat'
  }
} as const;

export function getProviderDisplayName(providerId: ProviderType): string {
  if (!providerId || !PROVIDER_CONFIG[providerId]) {
    console.warn(`Unknown provider: ${providerId}, falling back to default name`);
    return String(providerId || 'Unknown');
  }
  return PROVIDER_CONFIG[providerId].displayName;
} 