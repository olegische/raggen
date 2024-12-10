'use client';

import { useEffect, useState } from 'react';
import { ProviderType } from '../config/providers';

interface ModelSelectorProps {
  selectedModel: string;
  provider: ProviderType;
  onModelChange: (model: string) => void;
  disabled?: boolean;
  isLoading?: boolean;
}

export default function ModelSelector({
  selectedModel,
  provider,
  onModelChange,
  disabled = false,
  isLoading = false
}: ModelSelectorProps) {
  const [models, setModels] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadModels() {
      try {
        const response = await fetch(`/api/models?provider=${provider}`);
        if (!response.ok) throw new Error('Failed to load models');
        const data = await response.json();
        setModels(data);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      }
    }

    loadModels();
  }, [provider]);

  if (error) {
    return (
      <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
        <span>Ошибка: {error}</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-4 text-sm">
      <label htmlFor="model" className="text-gray-600 dark:text-gray-400">
        Модель:
      </label>
      <select
        id="model"
        value={selectedModel}
        onChange={(e) => onModelChange(e.target.value)}
        disabled={disabled || isLoading}
        className="px-2 py-1 border rounded bg-white dark:bg-gray-800 dark:border-gray-700 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {isLoading ? (
          <option value="">Загрузка...</option>
        ) : (
          models.map((model) => (
            <option key={model} value={model}>
              {model}
            </option>
          ))
        )}
      </select>
    </div>
  );
}
