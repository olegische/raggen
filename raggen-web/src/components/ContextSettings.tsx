import { useEffect, useState } from 'react';

interface ContextSettingsProps {
  maxContextMessages: number;
  contextScoreThreshold: number;
  contextEnabled: boolean;
  onSettingsChange: (settings: {
    maxContextMessages: number;
    contextScoreThreshold: number;
    contextEnabled: boolean;
  }) => void;
}

export default function ContextSettings({
  maxContextMessages,
  contextScoreThreshold,
  contextEnabled,
  onSettingsChange
}: ContextSettingsProps) {
  // Локальное состояние для контроля значений
  const [localSettings, setLocalSettings] = useState({
    maxContextMessages,
    contextScoreThreshold,
    contextEnabled
  });

  // Обновляем родительский компонент при изменении настроек
  useEffect(() => {
    onSettingsChange(localSettings);
  }, [localSettings, onSettingsChange]);

  return (
    <div className="p-4 bg-white dark:bg-gray-800 rounded-lg shadow">
      <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">
        Настройки контекста
      </h3>
      
      {/* Включение/выключение контекста */}
      <div className="mb-4">
        <label className="flex items-center space-x-2 cursor-pointer">
          <input
            type="checkbox"
            checked={localSettings.contextEnabled}
            onChange={(e) => setLocalSettings(prev => ({
              ...prev,
              contextEnabled: e.target.checked
            }))}
            className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
            aria-label="Включить контекстный поиск"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">
            Использовать контекстный поиск
          </span>
        </label>
      </div>

      {/* Максимальное количество сообщений */}
      <div className="mb-4">
        <label
          htmlFor="maxMessages"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
        >
          Максимальное количество сообщений
        </label>
        <div className="flex items-center space-x-2">
          <input
            type="range"
            id="maxMessages"
            min="1"
            max="10"
            value={localSettings.maxContextMessages}
            onChange={(e) => setLocalSettings(prev => ({
              ...prev,
              maxContextMessages: parseInt(e.target.value)
            }))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            aria-label="Максимальное количество сообщений для контекста"
          />
          <span className="text-sm text-gray-600 dark:text-gray-400 min-w-[2ch]">
            {localSettings.maxContextMessages}
          </span>
        </div>
      </div>

      {/* Порог релевантности */}
      <div className="mb-4">
        <label
          htmlFor="threshold"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
        >
          Минимальный порог релевантности
        </label>
        <div className="flex items-center space-x-2">
          <input
            type="range"
            id="threshold"
            min="0"
            max="100"
            value={localSettings.contextScoreThreshold * 100}
            onChange={(e) => setLocalSettings(prev => ({
              ...prev,
              contextScoreThreshold: parseInt(e.target.value) / 100
            }))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
            aria-label="Минимальный порог релевантности контекста"
          />
          <span className="text-sm text-gray-600 dark:text-gray-400 min-w-[3ch]">
            {Math.round(localSettings.contextScoreThreshold * 100)}%
          </span>
        </div>
      </div>

      {/* Подсказка */}
      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
        Эти настройки определяют, как система будет искать и использовать контекст из предыдущих сообщений
        для улучшения ответов.
      </p>
    </div>
  );
} 