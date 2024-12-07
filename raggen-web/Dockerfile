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

# Генерация Prisma Client с временным URL базы данных
ENV DATABASE_URL="file:/app/prisma/dev.db"
RUN npx prisma generate

# Копирование остальных файлов приложения
COPY . .

# Сборка приложения
RUN npm run build

# Открываем порт для Next.js
EXPOSE 3000

# Создаем директорию для базы данных и применяем миграции при запуске
CMD ["sh", "-c", "mkdir -p /app/prisma && npx prisma migrate deploy && npx prisma db push --accept-data-loss && npm start"] 