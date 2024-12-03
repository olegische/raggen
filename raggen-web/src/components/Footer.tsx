'use client';

import { useState } from 'react';
import { ProviderType } from '@/providers/factory';
import { GENERATION_CONFIG } from '@/config/generation';
import { getProviderDisplayName } from '@/config/providers';

interface FooterProps {
  provider: ProviderType;
  temperature: number;
  maxTokens: number;
  onSettingsChange: (settings: { temperature: number; maxTokens: number }) => void;
  disabled?: boolean;
}

export default function Footer({
  provider,
  temperature,
  maxTokens,
  onSettingsChange,
  disabled = false
}: FooterProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleTemperatureChange = (value: number) => {
    onSettingsChange({ temperature: value, maxTokens });
  };

  const handleMaxTokensChange = (value: number) => {
    onSettingsChange({ temperature, maxTokens: value });
  };

  return (
    <div className="sticky bottom-0 bg-white dark:bg-gray-900 border-t dark:border-gray-800">
      <div className="max-w-5xl mx-auto px-4 py-2">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white"
          disabled={disabled}
        >
          <span>Настройки {getProviderDisplayName(provider)}</span>
          <span className={`transform transition-transform ${isOpen ? 'rotate-180' : ''}`}>
            ▼
          </span>
        </button>

        {isOpen && (
          <div className="mt-4 space-y-4">
            <div>
              <label 
                htmlFor="temperature" 
                className="block text-sm font-medium text-gray-700 dark:text-gray-300"
              >
                Температура: {temperature}
              </label>
              <input
                id="temperature"
                type="range"
                min={GENERATION_CONFIG.temperature.min}
                max={GENERATION_CONFIG.temperature.max}
                step={GENERATION_CONFIG.temperature.step}
                value={temperature}
                onChange={(e) => handleTemperatureChange(parseFloat(e.target.value))}
                disabled={disabled}
                className="w-full mt-1"
              />
            </div>

            <div>
              <label 
                htmlFor="maxTokens" 
                className="block text-sm font-medium text-gray-700 dark:text-gray-300"
              >
                Максимум токенов: {maxTokens}
              </label>
              <input
                id="maxTokens"
                type="range"
                min={GENERATION_CONFIG.maxTokens.min}
                max={GENERATION_CONFIG.maxTokens.max}
                step={GENERATION_CONFIG.maxTokens.step}
                value={maxTokens}
                onChange={(e) => handleMaxTokensChange(parseInt(e.target.value))}
                disabled={disabled}
                className="w-full mt-1"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
} 