# Архитектура RAGGEN

## 1. Общий обзор
RAGGEN - это система для работы с большими языковыми моделями с поддержкой контекстного поиска на русском языке. Проект использует подход Retrieval-Augmented Generation (RAG) для улучшения качества ответов и организован как монорепозиторий, содержащий два основных сервиса.

## 2. Структура монорепозитория

```
raggen/
├── raggen-web/                    # Next.js веб-приложение и API
│   ├── src/
│   │   ├── app/                  # Next.js приложение
│   │   │   ├── api/             # API Routes
│   │   │   │   ├── chat/        # Обработка сообщений чата
│   │   │   │   │   └── route.ts # POST /api/chat
│   │   │   │   ├── messages/    # История сообщений
│   │   │   │   │   └── route.ts # GET /api/messages
│   │   │   │   ├── providers/   # API провайдеров
│   │   │   │   │   └── route.ts # GET /api/providers
│   │   │   │   └── models/      # API моделей
│   │   │   │       └── route.ts # GET /api/models
│   │   │   ├── fonts/          # Шрифты приложения
│   │   │   ├── globals.css     # Глобальные стили
│   │   │   ├── layout.tsx      # Корневой layout
│   │   │   └── page.tsx        # Главная страница
│   │   │
│   │   ├── components/         # React компоненты
│   │   │   ├── ChatWindow.tsx
│   │   │   ├── ContextIndicator.tsx
│   │   │   ├── ContextSettings.tsx
│   │   │   ├── Footer.tsx
│   │   │   ├── GenerationSettings.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── ModelSelector.tsx
│   │   │   ├── ProviderSelector.tsx
│   │   │   ├── ThemeProvider.tsx
│   │   │   └── ThemeToggle.tsx
│   │   │
│   │   ├── config/           # Конфигурация
│   │   │   ├── generation.ts
│   │   │   ├── prompts.ts
│   │   │   └── providers.ts
│   │   │
│   │   ├── lib/             # Общие утилиты
│   │   │   └── db.ts
│   │   │
│   │   ├── providers/       # Провайдеры LLM
│   │   │   ├── base.provider.ts
│   │   │   ├── factory.ts
│   │   │   ├── gigachat/
│   │   │   │   └── provider.ts
│   │   │   └── yandex/
│   │   │       └── provider.ts
│   │   │
│   │   ├── services/       # Сервисный слой
│   │   │   ├── chat.service.ts
│   │   │   ├── context.service.ts
│   │   │   ├── database.ts
│   │   │   ├── embed-api.ts
│   │   │   ├── model.service.ts
│   │   │   ├── prompt.service.ts
│   │   │   └── provider.service.ts
│   │   │
│   │   └── types/         # TypeScript типы
│   │
│   ├── prisma/           # База данных
│   │   ├── schema.prisma
│   │   └── migrations/
│   │
│   ├── tests/           # Тесты
│   │   └── __tests__/
│   │
│   └── public/          # Статические файлы
│
├── raggen-embed/        # Python сервис для эмбеддингов
│   ├── src/
│   │   ├── api/        # FastAPI endpoints
│   │   │   ├── embeddings.py
│   │   │   └── models.py
│   │   │
│   │   ├── core/      # Основная логика
│   │   │   ├── embeddings.py
│   │   │   └── vector_store/
│   │       ├── faiss_store.py
│   │       └── persistent_store.py
│   │   │
│   │   ├── config/    # Конфигурация
│   │   │   └── settings.py
│   │   │
│   │   └── utils/     # Утилиты
│   │       └── logging.py
│   │
│   ├── tests/         # Тесты
│   │   ├── test_api.py
│   │   ├── test_embeddings.py
│   │   └── test_vector_store.py
│   │
│   └── data/          # Данные
│       └── faiss/     # Индексы FAISS
│
├── data/              # Общие данные
│   ├── faiss/         # Индексы FAISS
│   └── sqlite/        # База данных SQLite
│
└── docs/             # Документация
    ├── architecture.md
    ├── backup.md
    ├── deployment.md
    └── rollback.md
```

## 3. Компоненты системы

### 3.1. raggen-web (Next.js)

#### 3.1.1. Frontend

##### Основные компоненты

###### ChatWindow
- **Назначение**: Основной компонент для отображения чата
- **Структура**:
  - UserMessage: Компонент сообщения пользователя
  - ProviderResponse: Компонент ответа от LLM
  - ContextIndicator: Индикатор использованного контекста
  - LoadingIndicator: Индикатор загрузки
  - EmptyState: Состояние пустого чата
  - ErrorState: Отображение ошибок
- **Функциональность**:
  - Отображение истории сообщений
  - Автоскроллинг к последнему сообщению
  - Копирование текста сообщений
  - Управление настройками контекста
  - Отображение метаданных (модель, провайдер)

###### Header
- **Назначение**: Управление основными настройками
- **Компоненты**:
  - ProviderSelector: Выбор LLM провайдера
  - ModelSelector: Выбор модели для выбранного провайдера
  - ThemeToggle: Переключение темы оформления
- **Функциональность**:
  - Динамическая загрузка доступных моделей
  - Сохранение настроек
  - Адаптивный дизайн

###### Footer
- **Назначение**: Ввод сообщений и управление генерацией
- **Компоненты**:
  - Input: Поле ввода сообщения
  - GenerationSettings: Настройки генерации текста
  - SendButton: Кнопка отправки
- **Функциональность**:
  - Отправка сообщений (Enter или кнопка)
  - Настройка параметров генерации
  - Индикация состояния загрузки

#### 3.1.2. Сервисный слой

##### ChatService
- **Назначение**: Центральный сервис обработки сообщений
- **Основные операции**:
  - Управление чатами и сообщениями
  - Интеграция с провайдерами LLM
  - Работа с контекстом
- **Ключевые методы**:
  ```typescript
  sendMessage(message: string, chatId?: string, options?: SendMessageOptions): Promise<{
    message: Message;
    context: ContextSearchResult[];
    chatId: string;
  }>
  ```

##### ContextService
- **Назначение**: Управление контекстным поиском
- **Функциональность**:
  - Поиск релевантного контекста
  - Векторизация сообщений
  - Сохранение использованного контекста
- **Взаимодействие**:
  - Интеграция с raggen-embed
  - Работа с FAISS индексами

##### PromptService
- **Назначение**: Форматирование промптов
- **Функциональность**:
  - Форматирование системных промптов
  - Добавление контекста
  - Управление историей сообщений

##### DatabaseService
- **Назначение**: Работа с базой данных
- **Операции**:
  - CRUD операции с чатами
  - Управление сообщениями
  - Сохранение эмбеддингов
  - Работа с контекстом

#### 3.1.3. Провайдеры LLM

##### BaseProvider
- **Назначение**: Абстрактный класс для провайдеров LLM
- **Интерфейс**:
  ```typescript
  interface GenerationOptions {
    model?: string;
    temperature?: number;
    maxTokens?: number;
  }
  ```

##### Реализации провайдеров
- **YandexGPTProvider**:
  - Поддержка моделей YandexGPT
  - Специфичные параметры генерации
  - Обработка ошибок API

- **GigaChatProvider**:
  - Интеграция с GigaChat API
  - Управление токенами
  - Специфичные промпты

##### ProviderFactory
- **Назначение**: Создание инстансов провайдеров
- **Функциональность**:
  - Динамическое создание провайдеров
  - Конфигурация параметров
  - Управление состоянием

#### 3.1.4. API Routes

##### /api/chat
- **Метод**: POST
- **Функциональность**:
  - Обработка сообщений
  - Интеграция с LLM
  - Сохранение контекста
- **Параметры**:
  ```typescript
  {
    message: string;
    provider: ProviderType;
    options: {
      model: string;
      temperature: number;
      maxTokens: number;
      maxContextMessages: number;
      contextScoreThreshold: number;
    }
  }
  ```

##### /api/messages
- **Метод**: GET
- **Функциональность**:
  - История сообщений
  - Фильтрация по чатам
  - Пагинация

##### /api/providers
- **Метод**: GET
- **Функциональность**:
  - Список доступных провайдеров
  - Статус провайдеров
  - Конфигурация

##### /api/models
- **Метод**: GET
- **Функциональность**:
  - Доступные модели
  - Параметры моделей
  - Зависимость от провайдера

### 3.2. raggen-embed (Python)

#### 3.2.1. API Layer
- **FastAPI** для REST endpoints
- Асинхронная обработка запросов
- Swagger документация

##### API Endpoints
- POST /api/v1/embed: Создание эмбеддинга
- POST /api/v1/embed/batch: Пакетное создание эмбеддингов
- POST /api/v1/search: Поиск похожих текстов

#### 3.2.2. Embedding Service
- Модель all-MiniLM-L6-v2 (384-мерные векторы)
- Кэширование модели
- Оптимизация памяти
- Обработка ошибок

#### 3.2.3. Vector Store
- FAISS для векторного поиска
- Персистентное хранение индексов
- Автоматическое резервное копирование
- Восстановление после сбоев

## 4. Взаимодействие компонентов

### 4.1. Обработка сообщений
1. Пользователь отправляет сообщение
2. ChatService обрабатывает запрос
3. ContextService ищет релевантный контекст через raggen-embed
4. PromptService формирует промпт с контекстом
5. Провайдер LLM генерирует о��вет
6. Ответ сохраняется и отображается пользователю

### 4.2. Контекстный поиск
1. Новое сообщение векторизуется через raggen-embed
2. Эмбеддинг схраняется в FAISS и БД
3. Выполняется поиск похожих сообщений
4. Контекст приоритизируется и фильтруется
5. Релевантный контекст добавляется в промпт

## 5. Безопасность

### 5.1. API
- Rate limiting
- Валидация входных данных
- CORS настройки
- Логирование запросов

### 5.2. Данные
- Безопасное хранение API ключей
- Шифрование конфигурации
- Изоляция компонентов через Docker

## 6. Мониторинг

### 6.1. Логирование
- Структурированные логи с request ID
- Уровни логирования
- Ротация логов

### 6.2. Метрики
- Latency API
- Использование памяти
- Размер индексов
- Статистика запросов

## 7. Требования к окружению

### 7.1. Системные требования
- Python 3.11+
- Node.js 18+
- Docker и Docker Compose
- Git

### 7.2. Ресурсы
- CPU: 2+ cores
- RAM: 4GB+
- Диск: 20GB+
- Сеть: 100Mbps+

### 7.3. Зависимости
```json
// raggen-web
{
  "dependencies": {
    "@prisma/client": "^5.22.0",
    "axios": "^1.6.2",
    "next": "15.0.3",
    "react": "^18.2.0",
    "zod": "^3.x"
  }
}

// raggen-embed
{
  "dependencies": {
    "fastapi": "^0.109.2",
    "sentence-transformers": "^2.3.1",
    "faiss-cpu": "^1.7.4"
  }
}
```