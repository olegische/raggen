# RAGGEN - Мультимодальный RAG на русском языке

RAGGEN - это система для работы с большими языковыми моделями с поддержкой контекстного поиска на русском языке. Проект состоит из двух основных сервисов: веб-интерфейса чата (raggen-web) и сервиса эмбеддингов (raggen-embed).

## Обзор системы

### raggen-web
Веб-интерфейс для взаимодействия с языковыми моделями:
- Поддержка различных LLM провайдеров (Yandex GPT, GigaChat)
- Контекстный поиск в истории сообщений
- Настройка параметров генерации
- Управление контекстом
- Светлая/тёмная тема

### raggen-embed
Сервис для работы с векторными представлениями текста:
- Генерация эмбеддингов с помощью all-MiniLM-L6-v2
- Векторное хранилище на базе FAISS
- REST API для векторных операций
- Кэширование моделей
- Оптимизация памяти

## Начало работы

### Предварительные требования
- Python 3.11+
- Node.js 18+
- Docker и Docker Compose
- Git

### Локальная разработка

1. Клонирование репозитория:
```bash
git clone https://github.com/olegische/raggen.git
cd raggen
```

2. Настройка переменных окружения:
```bash
# Основной .env файл
cp .env.example .env
# Настройка версии
./set-version.sh

# Конфигурация raggen-web
cp raggen-web/.env.example raggen-web/.env
# Настройте API ключи в raggen-web/.env

# Конфигурация raggen-embed
cp raggen-embed/.env.example raggen-embed/.env
```

3. Запуск с помощью Docker Compose:
```bash
docker-compose up -d
```

### Развертывание в production
Подробная инструкция по развертыванию на Debian 11 доступна в [документации по деплою](docs/deployment.md).

## Конфигурация

### raggen-web
- Настройка провайдеров LLM (API ключи)
- Параметры базы данных
- Настройки интерфейса

### raggen-embed
- Параметры векторного хранилища
- Настройки кэширования
- Конфигурация API

## API

### raggen-web
- REST API для работы с чатом
- WebSocket для real-time обновлений
- Документация: /api/docs

### raggen-embed
- REST API для работы с эмбеддингами
- Swagger UI: /docs
- ReDoc: /redoc

## Разработка

### Структура проекта
```
raggen/
├── raggen-web/         # Веб-интерфейс
├── raggen-embed/       # Сервис эмбеддингов
└── docs/              # Документация
```

### Тестирование
```bash
# Тестирование raggen-embed
cd raggen-embed
pytest

# Тестирование raggen-web
cd raggen-web
npm test
```

## Документация
- [Архитектура проекта](docs/architecture.md)
- [План релизов](ROADMAP.md)
- [Описание API](docs/app-description.md)
- [Схема базы данных](docs/database-schema.md)
- [Конфигурация сервисов](docs/configuration.md)
- [Примеры использования](docs/examples.md)
- [Инструкция по развертыванию](docs/deployment.md)

## Лицензия
MIT

## Авторы
- [Oleg Romanchuk](https://github.com/olegische) 