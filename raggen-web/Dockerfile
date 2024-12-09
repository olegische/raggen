# Явно указываем платформу
FROM --platform=linux/amd64 debian:11

# Установка необходимых пакетов
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Установка Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# Создание рабочей директории
WORKDIR /app

# Копирование package.json и package-lock.json
COPY package*.json ./

# Установка зависимостей
RUN npm install

# Копирование Prisma схемы и настройка базы данных
COPY prisma ./prisma/

# Генерация Prisma Client
RUN npx prisma generate

# Копирование остальных файлов приложения
COPY . .

# Сборка приложения
RUN npm run build

# Создаем директорию для базы данных и миграций
RUN mkdir -p /app/prisma/migrations

# Открываем порт для Next.js
EXPOSE 3000

# Запускаем приложение
CMD ["sh", "-c", "npx prisma migrate deploy && npx prisma db push --accept-data-loss && npm start"] 