# Развертывание RAGGEN в Production

## Предварительные требования

1. Образы должны быть собраны и опубликованы в registry из тестовой среды:
   - cr.yandex/${REGISTRY_ID}/raggen-web:${VERSION}
   - cr.yandex/${REGISTRY_ID}/raggen-embed:${VERSION}
   
См. [Инструкции по сборке](deployment-test.md) для деталей.

## Архитектура развертывания

В production среде сервисы развертываются на отдельных виртуальных машинах:
- VM1: raggen-web (Next.js приложение)
- VM2: raggen-embed (FastAPI сервис)

## Подготовка машин

На каждой машине выполните:

```bash
sudo apt-get update && sudo apt-get install -y \
    docker.io \
    docker-compose \
    nginx \
    git \
    fail2ban \
    vim \
    curl
```

## Настройка доступа к registry

На каждой машине настройте доступ к container registry:

```bash
# Для Yandex Container Registry
yc container registry configure-docker

# Или используя docker login
docker login cr.yandex
```

## Развертывание raggen-web (VM1)

1. Клонирование репозитория:
```bash
sudo git clone https://github.com/olegische/raggen.git /opt/raggen
cd /opt/raggen
```

2. Настройка переменных окружения:
```bash
# Основной .env файл
cp .env.example .env
./set-version.sh

# Создание директорий
sudo mkdir -p /opt/raggen/raggen-web
sudo mkdir -p /opt/raggen/data/sqlite/prisma
sudo chown -R $USER:$USER /opt/raggen/raggen-web /opt/raggen/data

# Копирование и настройка .env для web сервиса
cp raggen-web/.env.example /opt/raggen/raggen-web/.env
# Настройте API ключи в /opt/raggen/raggen-web/.env

# Важно: укажите правильный URL для сервиса эмбеддингов
# Измените значение NEXT_PUBLIC_EMBED_API_URL в .env
```

3. Настройка Nginx:
```bash
# Копирование конфигурации
sudo cp docs/tech/nginx-web.conf /etc/nginx/sites-available/raggen-web

# Настройка домена
sudo sed -i 's/your_domain.com/web.your-domain.com/' /etc/nginx/sites-available/raggen-web

# Активация конфигурации
sudo ln -sf /etc/nginx/sites-available/raggen-web /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

4. Запуск сервиса:

Через docker-compose:
```bash
# Pull образа из registry
docker-compose -f docker-compose.web.yml pull

# Запуск контейнера
docker-compose -f docker-compose.web.yml up -d
```

Или через docker run:
```bash
sudo docker run -d \
  --name raggen-web \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /opt/raggen/raggen-web/.env:/app/.env \
  -v /opt/raggen/data/sqlite/prisma:/app/prisma \
  cr.yandex/${REGISTRY_ID}/raggen-web:${VERSION}
```

## Развертывание raggen-embed (VM2)

1. Клонирование репозитория:
```bash
sudo git clone https://github.com/olegische/raggen.git /opt/raggen
cd /opt/raggen
```

2. Настройка переменных окружения:
```bash
# Основной .env файл
cp .env.example .env
./set-version.sh

# Создание директории для данных
sudo mkdir -p /opt/raggen/data/faiss
sudo chown -R $USER:$USER /opt/raggen/data

# Копирование и настройка .env для embed сервиса
cp raggen-embed/.env.example /opt/raggen/raggen-embed/.env

# Важно: настройте CORS для разрешения запросов от веб-сервера
# Добавьте домен веб-сервера в CORS_ORIGINS в /opt/raggen/raggen-embed/.env
```

3. Настройка Nginx:
```bash
# Копирование конфигурации
sudo cp docs/tech/nginx-embed.conf /etc/nginx/sites-available/raggen-embed

# Настройка домена
sudo sed -i 's/your_domain.com/embed.your-domain.com/' /etc/nginx/sites-available/raggen-embed

# Активация конфигурации
sudo ln -sf /etc/nginx/sites-available/raggen-embed /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

4. Запуск сервиса:

Через docker-compose:
```bash
# Pull образа из registry
docker-compose -f docker-compose.embed.yml pull

# Запуск контейнера
docker-compose -f docker-compose.embed.yml up -d
```

Или через docker run:
```bash
sudo docker run -d \
  --name raggen-embed \
  --restart unless-stopped \
  -p 8001:8001 \
  -v /opt/raggen/raggen-embed/.env:/app/.env \
  -v /opt/raggen/data/faiss:/app/data/faiss \
  -e HOST=0.0.0.0 \
  -e PORT=8001 \
  -e LOG_LEVEL=INFO \
  -e CORS_ORIGINS='["http://web.your-domain.com"]' \
  cr.yandex/${REGISTRY_ID}/raggen-embed:${VERSION}
```

## Проверка развертывания

### На VM1 (raggen-web):

1. Проверка статуса:
```bash
docker ps | grep raggen-web
docker logs raggen-web
```

2. Проверка логов Nginx:
```bash
sudo tail -f /var/log/nginx/error.log
```

### На VM2 (raggen-embed):

1. Проверка статуса:
```bash
docker ps | grep raggen-embed
docker logs raggen-embed
```

2. Проверка логов Nginx:
```bash
sudo tail -f /var/log/nginx/embed-error.log
```

## Обновление сервисов

### На VM1 (raggen-web):

```bash
cd /opt/raggen
sudo git pull
./set-version.sh

# При использовании docker-compose
docker-compose -f docker-compose.web.yml pull
docker-compose -f docker-compose.web.yml up -d

# Или при использовании docker run
docker pull cr.yandex/${REGISTRY_ID}/raggen-web:${VERSION}
docker stop raggen-web
docker rm raggen-web
docker run -d \
  --name raggen-web \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /opt/raggen/raggen-web/.env:/app/.env \
  -v /opt/raggen/data/sqlite/prisma:/app/prisma \
  cr.yandex/${REGISTRY_ID}/raggen-web:${VERSION}
```

### На VM2 (raggen-embed):

```bash
cd /opt/raggen
sudo git pull
./set-version.sh

# При использовании docker-compose
docker-compose -f docker-compose.embed.yml pull
docker-compose -f docker-compose.embed.yml up -d

# Или при использовании docker run
docker pull cr.yandex/${REGISTRY_ID}/raggen-embed:${VERSION}
docker stop raggen-embed
docker rm raggen-embed
docker run -d \
  --name raggen-embed \
  --restart unless-stopped \
  -p 8001:8001 \
  -v /opt/raggen/raggen-embed/.env:/app/.env \
  -v /opt/raggen/data/faiss:/app/data/faiss \
  -e HOST=0.0.0.0 \
  -e PORT=8001 \
  -e LOG_LEVEL=INFO \
  -e CORS_ORIGINS='["http://web.your-domain.com"]' \
  cr.yandex/${REGISTRY_ID}/raggen-embed:${VERSION}
```

## Важные замечания

1. **Сетевая конфигурация**:
   - Убедитесь, что порты 80 и 443 открыты на VM1
   - Убедитесь, что порт 8001 открыт на VM2
   - Настройте файрвол для разрешения трафика только между VM1 и VM2
   - Перед запуском embed сервиса проверьте, не занят ли порт 8001:
     ```bash
     # Проверка использования порта
     sudo netstat -tulpn | grep :8001
     # Если занят nginx
     sudo systemctl stop nginx
     # Освобождение порта от контейнера
     docker stop $(docker ps -q --filter "publish=8001")
     docker rm $(docker ps -aq --filter "publish=8001")
     ```

2. **SSL/TLS**:
   - Рекомендуется настроить HTTPS на обоих серверах
   - Используйте certbot для получения SSL-сертификатов
   - Обновите конфигурации Nginx для поддержки HTTPS

3. **Мониторинг**:
   - Настройте мониторинг на обоих серверах
   - Отслеживайте использование ресурсов
   - Настройте алерты для критических событий

4. **Резервное копирование**:
   - Регулярно создавайте резервные копии FAISS индексов на VM2
   - Регулярно создавайте резервные копии SQLite базы на VM1
   - Храните резервные копии в безопасном месте

5. **Безопасность**:
   - Используйте файрвол (UFW или iptables)
   - Настройте fail2ban на обоих серверах
   - Регулярно обновляйте системные пакеты
   - Используйте сильные пароли и SSH-ключи

6. **Container Registry**:
   - Убедитесь в актуальности учетных данных для registry
   - Регулярно обновляйте токены доступа
   - Проверяйте права доступа к registry
