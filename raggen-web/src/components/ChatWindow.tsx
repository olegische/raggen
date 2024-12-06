'use client';

import { useEffect, useRef } from 'react';
import { Message } from '@prisma/client';
import { ProviderType } from '@/providers/factory';
import { getProviderDisplayName, PROVIDER_CONFIG } from '@/config/providers';
import { ContextSearchResult } from '@/services/context.service';
import ContextIndicator from './ContextIndicator';

interface ChatWindowProps {
  messages: Message[];
  provider: ProviderType;
  loading?: boolean;
  error?: string | null;
  messageContexts?: Record<number, ContextSearchResult[]>;
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
  messageContexts
}: ChatWindowProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (typeof window !== 'undefined' && bottomRef.current?.scrollIntoView) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  return (
    <div className="flex-1 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-900">
      {error ? (
        <ErrorState error={error} />
      ) : messages.length === 0 ? (
        <EmptyState />
      ) : (
        <MessageList 
          messages={messages} 
          currentProvider={provider} 
          contexts={messageContexts}
        />
      )}
      {loading && <LoadingIndicator provider={provider} />}
      <div ref={bottomRef} />
    </div>
  );
} 