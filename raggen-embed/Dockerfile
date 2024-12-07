# Явно указываем платформу
FROM --platform=linux/amd64 python:3.11-slim

# Установка необходимых системных пакетов
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .
COPY pyproject.toml .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Создание директории для FAISS индексов
RUN mkdir -p /app/data/faiss

# Копирование исходного кода
COPY src ./src
COPY .env.example .env

# Установка переменных окружения
ENV PYTHONPATH=/app/src
ENV FAISS_INDEX_PATH=/app/data/faiss/index.faiss

# Открываем порт для FastAPI
EXPOSE 8001

# Запуск приложения
CMD ["python", "src/run.py"] 