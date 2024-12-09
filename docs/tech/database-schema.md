# Схема базы данных RAGGEN

## Обзор

База данных RAGGEN построена на SQLite и управляется через Prisma ORM. Схема оптимизирована для работы с чатами, сообщениями и их векторными представлениями (эмбеддингами).

## Модели

### Chat

Представляет собой чат-сессию с определенным провайдером LLM.

| Поле | Тип | Описание | Индекс |
|------|-----|----------|---------|
| id | String | Уникальный UUID чата | Primary Key |
| provider | String | Тип провайдера (yandex/gigachat) | ✓ |
| createdAt | DateTime | Время создания чата | - |
| updatedAt | DateTime | Время последнего обновления | - |

Связи:
- `messages`: Один-ко-многим с моделью Message

### Message

Хранит сообщения пользователя и ответы модели.

| Поле | Тип | Описание | Индекс |
|------|-----|----------|---------|
| id | Int | Уникальный ID сообщения | Primary Key |
| chatId | String | ID чата | ✓ |
| message | String | Текст сообщения пользователя | - |
| response | String? | Ответ от модели (опционально) | - |
| model | String | Используемая модель | - |
| provider | String | Провайдер, сгенерировавший ответ | - |
| timestamp | DateTime | Время создания сообщения | ✓ |
| temperature | Float | Параметр температуры генерации | - |
| maxTokens | Int | Максимальное количество токенов | - |

Связи:
- `chat`: Многие-к-одному с моделью Chat
- `embedding`: Один-к-одному с моделью Embedding
- `usedContext`: Один-ко-многим с моделью Context

### Embedding

Хранит векторные представления сообщений.

| Поле | Тип | Описание | Индекс |
|------|-----|----------|---------|
| id | Int | Уникальный ID эмбеддинга | Primary Key |
| messageId | Int | ID связанного сообщения | Unique |
| vector | Bytes | Эмбеддинг в бинарном формате (384 * 4 байта) | - |
| vectorId | Int | ID вектора в FAISS | ✓ |
| createdAt | DateTime | Время создания эмбеддинга | - |

Связи:
- `message`: Один-к-одному с моделью Message

### Context

Хранит информацию об использованном контексте в сообщениях.

| Поле | Тип | Описание | Индекс |
|------|-----|----------|---------|
| id | Int | Уникальный ID контекста | Primary Key |
| messageId | Int | ID сообщения, использующего контекст | ✓ |
| sourceId | Int | ID исходного сообщения-контекста | ✓ |
| score | Float | Оценка релевантности (0-1) | ✓ |
| usedInPrompt | Boolean | Флаг использования в промпте | - |
| createdAt | DateTime | Время создания записи | - |

Связи:
- `message`: Многие-к-одному с моделью Message

## Индексы

Схема включает следующие индексы для оптимизации запросов:

1. Chat:
   - `provider`: Для быстрого поиска чатов по провайдеру

2. Message:
   - `chatId`: Для быстрой выборки сообщений чата
   - `timestamp`: Для сортировки и фильтрации по времени

3. Embedding:
   - `vectorId`: Для быстрого поиска по ID вектора в FAISS

4. Context:
   - `messageId`: Для поиска контекста сообщения
   - `sourceId`: Для поиска использований сообщения как контекста
   - `score`: Для фильтрации по релевантности

## Особенности реализации

1. **Бинарное хранение векторов**:
   - Векторы эмбеддингов хранятся в бинарном формате
   - Каждый вектор имеет размерность 384 (all-MiniLM-L6-v2)
   - Размер вектора: 384 * 4 = 1536 байт

2. **Связь с FAISS**:
   - `vectorId` в модели Embedding соответствует индексу в FAISS
   - Обеспечивает синхронизацию между БД и векторным хранилищем

3. **Оптимизация контекста**:
   - Индекс по score позволяет быстро находить наиболее релевантный контекст
   - Связь многие-ко-многим между сообщениями через таблицу Context

## Миграции

База данных использует Prisma Migrate для управления схемой:

```bash
# Создание миграции
npx prisma migrate dev --name init

# Применение миграций
npx prisma migrate deploy
```

## Рекомендации по работе

1. Используйте транзакции при создании связанных записей
2. Периодически очищайте устаревшие записи
3. Следите за размером бинарных данных эмбеддингов
4. Используйте индексы при составлении запросов 