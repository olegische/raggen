# Архитектура Multimodal RAG RU

## 1. Общий обзор
Multimodal RAG RU - это веб-приложение для взаимодействия с различными языковыми моделями (LLM), использующее подход Retrieval-Augmented Generation (RAG) для улучшения качества ответов. Проект организован как монорепозиторий, содержащий несколько сервисов.

## 2. Структура монорепозитория

```
raggen/
├── raggen-web/                    # Next.js веб-приложение и API
│   ├── src/
│   │   ├── app/                  # Next.js приложение
│   │   │   ├── api/             # API Routes
│   │   │   │   ├── chat/        # Обработка сообщений чата
│   │   │   │   │   └── route.ts
│   │   │   │   ├── messages/    # История сообщений
│   │   │   │   │   └── route.ts
│   │   │   │   ├── files/      # Загрузка файлов
│   │   │   │   │   └── route.ts
│   │   │   │   └── models/     # API моделей
│   │   │   │       └── route.ts
│   │   │   ├── fonts/          # Шрифты приложения
│   │   │   │   ├── GeistMonoVF.woff
│   │   │   │   └── GeistVF.woff
│   │   │   ├── favicon.ico
│   │   │   ├── globals.css     # Глобальные стили и Tailwind
│   │   │   ├── layout.tsx      # Корневой layout
│   │   │   └── page.tsx        # Главная страница
│   │   │
│   │   ├── components/         # React компоненты
│   │   │   ├── chat/          # Компоненты чата
│   │   │   │   ├── ChatWindow.tsx
│   │   │   │   ├── Message.tsx
│   │   │   │   └── MessageList.tsx
│   │   │   ├── ui/            # UI компоненты
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Input.tsx
│   │   │   │   └── Select.tsx
│   │   │   ├── settings/      # Компоненты настроек
│   │   │   │   ├── GenerationSettings.tsx
│   │   │   │   └── ModelSelector.tsx
│   │   │   ├── Footer.tsx
│   │   │   ├── Header.tsx
│   │   │   └── ThemeProvider.tsx
│   │   │
│   │   ├── providers/         # Провайдеры LLM
│   │   │   ├── base/
│   │   │   │   ├── provider.ts
│   │   │   │   └── types.ts
│   │   │   ├── yandex/
│   │   │   │   ├── provider.ts
│   │   │   │   ├── types.ts
│   │   │   │   └── utils.ts
│   │   │   ├── gigachat/
│   │   │   │   ├── provider.ts
│   │   │   │   ├── types.ts
│   │   │   │   └── utils.ts
│   │   │   └── factory.ts
│   │   │
│   │   ├── services/         # Сервисный слой
│   │   │   ├── chat.service.ts
│   │   │   ├── model.service.ts
│   │   │   ├── provider.service.ts
│   │   │   └── embed.service.ts
│   │   │
│   │   └── lib/            # Общие утилиты
│   │       ├── db.ts      # Инициализация Prisma
│   │       ├── config.ts  # Конфигурация
│   │       ├── errors.ts  # Обработка ошибок
│   │       └── utils.ts   # Вспомогательные функции
│   │
│   ├── prisma/            # Схема базы данных
│   │   ├── schema.prisma
│   │   └── migrations/
│   │
│   ├── public/           # Статические файлы
│   │   └── images/
│   │
│   ├── tests/           # Тесты
│   │   ├── unit/
│   │   └── integration/
│   │
│   ├── .env.example     # Пример переменных окружения
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.js
│
├── raggen-embed/         # Python сервис для работы с эмбеддингами
│   ├── src/
│   │   ├── api/         # FastAPI endpoints
│   │   │   ├── routes/
│   │   │   │   ├── embed.py
│   │   │   │   ├── search.py
│   │   │   │   └── chunks.py
│   │   │   ├── middleware/
│   │   │   │   └── auth.py
│   │   │   └── app.py
│   │   │
│   │   ├── embeddings/  # Логика работы с эмбеддингами
│   │   │   ├── service.py
│   │   │   ├── models.py
│   │   │   └── utils.py
│   │   │
│   │   ├── models/     # Модели данных
│   │   │   ├── document.py
│   │   │   ├── chunk.py
│   │   │   └── embedding.py
│   │   │
│   │   ├── storage/    # Работа с хранилищем
│   │   │   ├── vector_store.py
│   │   │   └── sqlite.py
│   │   │
│   │   └── utils/     # Утилиты
│   │       ├── text.py
│   │       ├── config.py
│   │       └── logger.py
│   │
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── test_embeddings.py
│   │   │   └── test_chunks.py
│   │   └── integration/
│   │       └── test_api.py
│   │
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   │
│   ├── requirements/
│   │   ├── base.txt
│   │   ├── dev.txt
│   │   └── prod.txt
│   │
│   ├── .env.example
│   ├── README.md
│   └── setup.py
│
└── docs/               # Документация проекта
    ├── architecture.md
    ├── api-docs.md
    └── deployment.md
```

## 3. Компоненты системы

### 3.1. raggen-web (Next.js)

#### 3.1.1. Frontend
- **Next.js** для серверного рендеринга
- **Tailwind CSS** для стилизации
- **React** компоненты для UI

##### Основные компоненты
- ChatWindow: Отображение истории сообщений
- Footer: Ввод сообщений, загрузка файлов контекста и настройки
- Header: Навигация и выбор моделей
- ModelSelector: Выбор провайдера и модели
- GenerationSettings: Настройки генерации текста

#### 3.1.2. Backend API
- **API Routes** для обработки запросов
- Интеграция с провайдерами LLM
- Взаимодействие с raggen-embed через REST API

##### API Endpoints
- POST /api/chat: Обработка сообщений чата
- GET /api/messages: История сообщений
- POST /api/files: Загрузка файлов
- GET /api/models: Доступные модели

#### 3.1.3. Провайдеры LLM
- YandexGPT Provider
- GigaChat Provider
- Абстрактный BaseProvider
- ProviderFactory для создания инстансов

#### 3.1.4. База данных
- **Prisma ORM**
- **SQLite** для хранения данных
- Схема данных:
  ```
  User (1) --- (n) Chat (1) --- (n) Message
  Document (1) --- (n) Chunk
  ```

### 3.2. raggen-embed (Python)

#### 3.2.1. API Layer
- **FastAPI** для REST endpoints
- Асинхронная обработка запросов
- Swagger документация

##### API Endpoints
- POST /api/embed: Создание эмбеддингов
- GET /api/search: Поиск похожих документов
- POST /api/chunks: Создание чанков из документов

#### 3.2.2. Embedding Service
- Генерация эмбеддингов с использованием модели all-MiniLM-L6-v2 (HuggingFace)
- Чанкинг документов
- Кэширование результатов

#### 3.2.3. Vector Store
- FAISS для хранения и поиска 384-мерных векторов (размерность all-MiniLM-L6-v2)
- Интеграция с SQLite для метаданных
- Оптимизация поиска

## 4. Взаимодействие компонентов

### 4.1. Процесс обработки запроса
1. Пользователь отправляет сообщение через UI
2. raggen-web обрабатывает запрос и ищет релевантный контекст
3. Запрос к raggen-embed для получения похожих документов
4. Объединение контекста с запросом пользователя
5. Отправка в LLM API (YandexGPT/GigaChat)
6. Возврат ответа пользователю

### 4.2. Процесс загрузки документов
1. Пользователь загружает документ
2. raggen-web сохраняет документ и отправляет в raggen-embed
3. raggen-embed разбивает на чанки и создает эмбеддинги
4. Эмбеддинги сохраняются в vector store
5. Метаданные сохраняются в SQLite

## 5. Безопасность

### 5.1. API Security
- Rate limiting
- Валидация входных данных
- CORS настройки
- Защита от SQL-инъекций через Prisma

### 5.2. Данные
- Безопасное хранение API ключей
- Шифрование чувствительных данных
- Логирование доступа

## 6. Развертывание

### 6.1. raggen-web
- Vercel для Next.js приложения
- Docker для API сервиса
- Nginx как reverse proxy

### 6.2. raggen-embed
- Docker контейнер
- Yandex Cloud / Sber Cloud
- Масштабирование через Docker Compose

## 7. Мониторинг

### 7.1. Логирование
- Структурированные логи
- Отслеживание ошибок
- Мониторинг производительности

### 7.2. Метрики
- Latency API endpoints
- Использование ресурсов
- Качество ответов LLM

## 8. Зависимости

### 8.1. raggen-web
```json
{
  "dependencies": {
    "next": "14.x",
    "react": "18.x",
    "prisma": "5.x",
    "tailwindcss": "3.x"
  }
}
```

### 8.2. raggen-embed
```python
sentence-transformers==2.2.2  # Для модели all-MiniLM-L6-v2
fastapi>=0.100.0
faiss-cpu>=1.7.4
pydantic>=2.0.0
```

## 9. Требования к окружению
- Node.js 20.x
- Python 3.11+
- Docker
- Git 