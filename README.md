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

#### Тестовая среда (для сборки)
- Python 3.11+
- Node.js 18+
- Docker и Docker Compose
- Git
- Минимум 4GB RAM для сборки образов
- Доступ к container registry

#### Production среда (для развертывания)
- Docker и Docker Compose
- Nginx
- Git
- Доступ к container registry

### Процесс развертывания

1. Сборка в тестовой среде:
```bash
# Клонирование репозитория
git clone https://github.com/olegische/raggen.git
cd raggen

# Настройка окружения
cp .env.example .env
./set-version.sh

# Сборка образов
docker-compose -f docker-compose.build.yml build

# Публикация образов
docker-compose -f docker-compose.build.yml push
```

2. Развертывание в production:
- Сервисы развертываются на отдельных серверах
- Используются готовые образы из registry
- Подробные инструкции в [документации по развертыванию](docs/tech/deployment-prod.md)

### Локальная разработка

1. Клонирование и настройка:
```bash
git clone https://github.com/olegische/raggen.git
cd raggen

cp .env.example .env
./set-version.sh

cp raggen-web/.env.example raggen-web/.env
cp raggen-embed/.env.example raggen-embed/.env
```

2. Запуск для разработки:
```bash
docker-compose up -d
```

## Документация

### [Пользовательская документация](docs/user/README.md)
- [Начало работы](docs/user/getting-started.md)
- [Основные функции](docs/user/features.md)
- [Настройка](docs/user/configuration.md)
- [Работа с контекстом](docs/user/context.md)
- [Провайдеры и модели](docs/user/providers.md)
- [FAQ](docs/user/faq.md)

### [Техническая документация](docs/tech/README.md)
- [Сборка и тестирование](docs/tech/deployment-test.md)
- [Production развертывание](docs/tech/deployment-prod.md)
- [Архитектура проекта](docs/tech/architecture.md)
- [Схема базы данных](docs/tech/database-schema.md)
- [Конфигурация сервисов](docs/tech/configuration.md)
- [Настройка окружения](docs/tech/environment.md)
- [Процедуры резервного копирования](docs/tech/backup.md)
- [Процедуры отката](docs/tech/rollback.md)

## Лицензия
MIT

## Авторы
- [Oleg Romanchuk](https://github.com/olegische)
