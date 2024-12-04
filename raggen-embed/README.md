# Raggen Embed Service

Сервис для генерации эмбеддингов текста с использованием модели sentence-transformers.

## Установка

### С использованием Poetry (рекомендуется)

```bash
poetry install
```

### С использованием pip

```bash
pip install -r requirements.txt
```

## Настройка

1. Скопируйте `.env.example` в `.env`:

```bash
cp .env.example .env
```

2. Настройте параметры в `.env` файле

## Запуск

### В режиме разработки

```bash
cd src
python run.py
```

### В продакшн режиме

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8001
```

## API Endpoints

### Health Check
```
GET /health
```

### Создание эмбеддинга для одного текста
```
POST /api/v1/embeddings/embed
Content-Type: application/json

{
    "text": "Your text here"
}
```

### Создание эмбеддингов для нескольких текстов
```
POST /api/v1/embeddings/embed-batch
Content-Type: application/json

{
    "texts": ["First text", "Second text", ...]
}
```

## Тестирование

```bash
pytest
```

## Структура проекта

```
raggen-embed/
├── src/
│   ├── api/          # API endpoints
│   ├── core/         # Core business logic
│   ├── config/       # Configuration
│   ├── models/       # Data models
│   ├── services/     # Services
│   └── utils/        # Utilities
├── tests/            # Tests
├── pyproject.toml    # Poetry dependencies
└── requirements.txt  # Pip dependencies
``` 