# Конфигурация сервисов RAGGEN

## Обзор

RAGGEN использует конфигурацию на основе переменных окружения для обоих сервисов. Все настройки могут быть определены через `.env` файлы или переменные окружения системы.

## raggen-web

### Основные настройки

| Переменная | Описание | Значение по умолчанию | Обязательная |
|------------|----------|----------------------|--------------|
| NODE_ENV | Окружение (development/production) | development | Нет |
| PORT | Порт для Next.js приложения | 3000 | Нет |
| DATABASE_URL | URL для SQLite базы данных | file:./prisma/dev.db | Да |
| NEXT_PUBLIC_EMBED_API_URL | URL сервиса эмбеддингов | http://raggen-embed:8001 | Да |

### Настройки Yandex GPT

| Переменная | Описание | Обязательная |
|------------|----------|--------------|
| YANDEX_GPT_API_URL | URL API Yandex GPT | Да |
| YANDEX_API_KEY_ID | ID API ключа | Да |
| YANDEX_API_KEY | API ключ | Да |
| YANDEX_FOLDER_ID | ID каталога в Yandex Cloud | Да |

### Настройки GigaChat

| Переменная | Описание | Обязательная |
|------------|----------|--------------|
| GIGACHAT_API_URL | URL API GigaChat | Да |
| GIGACHAT_CREDENTIALS | Авторизационный ключ | Да |
| GIGACHAT_SCOPE | Область доступа | Да |
| GIGACHAT_VERIFY_SSL_CERTS | Проверка SSL сертификатов | Нет |

## raggen-embed

### API настройки

| Переменная | Описание | Значение по умолчанию | Обязательная |
|------------|----------|----------------------|--------------|
| API_TITLE | Название API | Raggen Embed API | Нет |
| API_DESCRIPTION | Описание API | API for text embeddings and vector search | Нет |
| API_VERSION | Версия API | 1.0.0 | Нет |

### Настройки модели

| Переменная | Описание | Значение по умолчанию | Обязательная |
|------------|----------|----------------------|--------------|
| MODEL_NAME | Название модели | sentence-transformers/all-MiniLM-L6-v2 | Нет |
| VECTOR_DIM | Размер��ость векторов | 384 | Нет |

### Настройки сервера

| Переменная | Описание | Значение по умолчанию | Обязательная |
|------------|----------|----------------------|--------------|
| HOST | Хост для привязки сервера | 0.0.0.0 | Нет |
| PORT | Порт для FastAPI | 8001 | Нет |

### CORS настройки

| Переменная | Описание | Значение по умолчанию | Обязательная |
|------------|----------|----------------------|--------------|
| CORS_ORIGINS | Разрешенные источники | ["*"] | Нет |
| CORS_ALLOW_CREDENTIALS | Разрешить credentials | true | Нет |
| CORS_ALLOW_METHODS | Разрешенные методы | ["*"] | Нет |
| CORS_ALLOW_HEADERS | Разрешенные заголовки | ["*"] | Нет |

### Настройки производительности

| Переменная | Описание | Значение по умолчанию | Обязательная |
|------------|----------|----------------------|--------------|
| BATCH_SIZE | Размер батча для обработки | 32 | Нет |
| MAX_TEXT_LENGTH | Максимальная длина текста | 512 | Нет |
| REQUEST_TIMEOUT | Таймаут запроса (сек) | 30 | Нет |

### Настройки логирования

| Переменная | Описание | Значение по умолчанию | Обязательная |
|------------|----------|----------------------|--------------|
| LOG_LEVEL | Уровень логирования | INFO | Нет |
| LOG_FORMAT | Формат сообщений лога | %(asctime)s [%(levelname)s] [%(name)s] [%(request_id)s] %(message)s | Нет |

### Настройки FAISS

| Переменная | Описание | Значение по умолчанию | Обязательная |
|------------|----------|----------------------|--------------|
| N_CLUSTERS | Количество кластеров для IVF | 100 | Нет |
| N_PROBE | Количество проверяемых кластеров | 10 | Нет |
| N_RESULTS | Количество результатов по умолчанию | 5 | Нет |
| FAISS_INDEX_PATH | Путь к индексу FAISS | data/faiss/index.faiss | Да |

## Примеры конфигурации

### raggen-web (.env)

```env
# Основные настройки
NODE_ENV=development
PORT=3000
DATABASE_URL="file:./prisma/dev.db"
NEXT_PUBLIC_EMBED_API_URL=http://raggen-embed:8001

# Yandex GPT
YANDEX_GPT_API_URL=https://llm.api.cloud.yandex.net/foundationModels/v1/completion
YANDEX_API_KEY_ID=your_api_key_id
YANDEX_API_KEY=your_api_key
YANDEX_FOLDER_ID=your_folder_id

# GigaChat
GIGACHAT_API_URL=https://gigachat.devices.sberbank.ru/api/v1
GIGACHAT_CREDENTIALS=your_auth_key
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_VERIFY_SSL_CERTS=false
```

### raggen-embed (.env)

```env
# API настройки
API_TITLE=Raggen Embed API
API_VERSION=1.0.0

# Настройки сервера
HOST=0.0.0.0
PORT=8001

# CORS
CORS_ORIGINS=["http://localhost:3000"]
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=["GET", "POST"]
CORS_ALLOW_HEADERS=["*"]

# Производительность
BATCH_SIZE=32
MAX_TEXT_LENGTH=512
REQUEST_TIMEOUT=30

# Логирование
LOG_LEVEL=INFO

# FAISS
N_CLUSTERS=100
N_PROBE=10
N_RESULTS=5
FAISS_INDEX_PATH=/app/data/faiss/index.faiss
```

## Рекомендации по безопасности

1. **Секреты**:
   - Не коммитьте `.env` файлы в репозиторий
   - Используйте `.env.example` для примеров конфигурации
   - В production используйте секреты Docker или системы оркестрации

2. **CORS**:
   - В production точно определите разрешенные источники
   - Не используйте wildcard (*) для origins
   - Ограничьте список разрешенных методов и заголовков

3. **Логирование**:
   - В production установите уровень логирования INFO или WARNING
   - Настройте ротацию логов
   - Не логируйте чувствительные данные

4. **Таймауты**:
   - Настройте разумные таймауты для предотвращения DOS
   - Учитывайте время обработки больших текстов
   - Добавьте rate limiting в production

## Переопределение настроек

### В Docker Compose

```yaml
services:
  raggen-embed:
    environment:
      - HOST=0.0.0.0
      - PORT=8001
      - LOG_LEVEL=INFO
    
  raggen-web:
    environment:
      - NODE_ENV=production
      - PORT=3000
      - DATABASE_URL=file:/app/prisma/prod.db
```

### В командной строке

```bash
# raggen-embed
export LOG_LEVEL=DEBUG
export PORT=8002

# raggen-web
export NODE_ENV=development
export PORT=3001
``` 