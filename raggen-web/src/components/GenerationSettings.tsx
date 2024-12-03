'use client';

interface GenerationSettingsProps {
  temperature: number;
  maxTokens: number;
  onSettingsChange: (settings: { temperature: number; maxTokens: number }) => void;
  disabled?: boolean;
}

export default function GenerationSettings({
  temperature,
  maxTokens,
  onSettingsChange,
  disabled = false
}: GenerationSettingsProps) {
  return (
    <div className="flex gap-4 items-center text-sm text-gray-600 dark:text-gray-400">
      <div className="flex items-center gap-2">
        <label htmlFor="temperature">Температура:</label>
        <input
          id="temperature"
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={temperature}
          onChange={(e) => 
            onSettingsChange({ 
              temperature: parseFloat(e.target.value), 
              maxTokens 
            })
          }
          disabled={disabled}
          className="w-24 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
        />
        <span className="w-8">{temperature}</span>
      </div>

      <div className="flex items-center gap-2">
        <label htmlFor="maxTokens">Токены:</label>
        <input
          id="maxTokens"
          type="number"
          min="1"
          max="2000"
          value={String(maxTokens)}
          onChange={(e) => 
            onSettingsChange({ 
              temperature, 
              maxTokens: parseInt(e.target.value) 
            })
          }
          disabled={disabled}
          className="w-20 px-2 py-1 border rounded dark:bg-gray-800 dark:border-gray-700"
        />
      </div>
    </div>
  );
} 