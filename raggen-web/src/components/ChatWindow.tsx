'use client';

import { useEffect, useRef, useState } from 'react';
import { Message } from '@prisma/client';
import { ProviderType } from '@/providers/factory';
import { getProviderDisplayName, PROVIDER_CONFIG } from '@/config/providers';
import { ContextSearchResult } from '@/services/context.service';
import ContextIndicator from './ContextIndicator';
import ContextSettings from './ContextSettings';

interface ChatWindowProps {
  messages: Message[];
  provider: ProviderType;
  loading?: boolean;
  error?: string | null;
  messageContexts?: Record<number, ContextSearchResult[]>;
  onContextSettingsChange?: (settings: {
    maxContextMessages: number;
    contextScoreThreshold: number;
    contextEnabled: boolean;
  }) => void;
  contextSettings?: {
    maxContextMessages: number;
    contextScoreThreshold: number;
    contextEnabled: boolean;
  };
}

// Компонент кнопки копирования
const CopyButton = ({ text }: { text: string }) => {
  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch (err) {
      console.error('Failed to copy text:', err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-gray-100 dark:hover:bg-gray-700 rounded"
      title="Копировать текст"
    >
      <svg
        xmlns="http://www.w3.org/2000/svg"
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-gray-500 dark:text-gray-400"
      >
        <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
        <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
      </svg>
    </button>
  );
};

// Компонент сообщения пользователя
const UserMessage = ({ message }: { message: string }) => (
  <div className="flex justify-end mb-4">
    <div className="inline-block max-w-[80%] p-4 rounded-lg bg-blue-500 text-white group">
      <div className="flex items-center justify-between mb-1">
        <div className="text-sm">Вы</div>
        <CopyButton text={message} />
      </div>
      <div className="whitespace-pre-wrap break-words" data-testid="user-message">
        {message}
      </div>
    </div>
  </div>
);

// Компонент ответа провайдера
const ProviderResponse = ({ 
  response, 
  model, 
  provider, 
  currentProvider,
  context
}: { 
  response: string; 
  model?: string | null; 
  provider: string; 
  currentProvider: ProviderType;
  context?: ContextSearchResult[];
}) => {
  // Используем currentProvider как fallback, если provider не является валидным ProviderType
  const displayProvider = (PROVIDER_CONFIG[provider as ProviderType] ? provider : currentProvider) as ProviderType;
  
  return (
    <div className="flex justify-start">
      <div className="inline-block max-w-[80%] p-4 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 group">
        <div className="text-sm mb-1">{getProviderDisplayName(displayProvider)}</div>
        <div className="whitespace-pre-wrap break-words" data-testid="provider-response">
          {response}
        </div>
        <div className="mt-2 flex items-center justify-between">
          {model && (
            <div className="text-xs opacity-70">
              Модель: {model}
            </div>
          )}
          <CopyButton text={response} />
        </div>
        {context && <ContextIndicator context={context} />}
      </div>
    </div>
  );
};

// Компонент сообщения с ответом
const MessageWithResponse = ({ 
  message, 
  response, 
  model, 
  provider, 
  currentProvider,
  messageId,
  contexts
}: { 
  message: string; 
  response?: string | null; 
  model?: string | null; 
  provider: string; 
  currentProvider: ProviderType;
  messageId: number;
  contexts?: Record<number, ContextSearchResult[]>;
}) => (
  <div className="space-y-4">
    <UserMessage message={message} />
    {response && (
      <ProviderResponse 
        response={response} 
        model={model} 
        provider={provider}
        currentProvider={currentProvider}
        context={contexts?.[messageId]}
      />
    )}
  </div>
);

// Компонент загрузки
const LoadingIndicator = ({ provider }: { provider: ProviderType }) => (
  <div className="flex justify-center items-center py-4">
    <div className="animate-pulse text-gray-500 dark:text-gray-400">
      {getProviderDisplayName(provider)} печатает...
    </div>
  </div>
);

// Компонент пустого состояния
const EmptyState = () => (
  <div className="text-center text-gray-500 dark:text-gray-400">
    Начните диалог, отправив сообщение
  </div>
);

// Компонент ошибки
const ErrorState = ({ error }: { error: string }) => (
  <div className="p-4 text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 rounded-lg" data-testid="error-message">
    {error}
  </div>
);

// Компонент списка сообщений
const MessageList = ({ 
  messages, 
  currentProvider,
  contexts 
}: { 
  messages: Message[]; 
  currentProvider: ProviderType;
  contexts?: Record<number, ContextSearchResult[]>;
}) => (
  <div className="space-y-4">
    {messages.map((msg) => (
      <MessageWithResponse 
        key={msg.id}
        message={msg.message}
        response={msg.response}
        model={msg.model}
        provider={msg.provider}
        currentProvider={currentProvider}
        messageId={msg.id}
        contexts={contexts}
      />
    ))}
  </div>
);

export default function ChatWindow({ 
  messages, 
  provider,
  loading = false,
  error = null,
  messageContexts,
  onContextSettingsChange,
  contextSettings = {
    maxContextMessages: 5,
    contextScoreThreshold: 0.7,
    contextEnabled: true
  }
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined' && bottomRef.current?.scrollIntoView) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="flex flex-col h-full">
      {/* Кнопка настроек */}
      <div className="flex justify-end p-2">
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          aria-label={showSettings ? 'Скрыть настройки контекста' : 'Показать настройки контекста'}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
          </svg>
        </button>
      </div>

      {/* Панель настроек */}
      {showSettings && onContextSettingsChange && (
        <div className="mb-4 mx-4">
          <ContextSettings
            maxContextMessages={contextSettings.maxContextMessages}
            contextScoreThreshold={contextSettings.contextScoreThreshold}
            contextEnabled={contextSettings.contextEnabled}
            onSettingsChange={onContextSettingsChange}
          />
        </div>
      )}

      {/* Существующий код чата */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 dark:text-gray-400">
            Начните диалог, отправив сообщение
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className="mb-4">
              <UserMessage message={message.message} />
              {message.response && (
                <ProviderResponse 
                  response={message.response} 
                  model={message.model} 
                  provider={message.provider}
                  currentProvider={provider}
                  context={messageContexts?.[message.id]}
                />
              )}
              {messageContexts?.[message.id] && (
                <ContextIndicator context={messageContexts[message.id]} />
              )}
            </div>
          ))
        )}
        {loading && (
          <div className="text-gray-500 dark:text-gray-400">
            {getProviderDisplayName(provider)} печатает...
          </div>
        )}
        {error && (
          <div className="text-red-500 dark:text-red-400">
            {error}
          </div>
        )}
      </div>
    </div>
  );
} 