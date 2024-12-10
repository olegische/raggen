# Развертывание RAGGEN на Debian 11

## Подготовка системы

1. Установка необходимых пакетов:
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

2. Настройка fail2ban:
```bash
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local
sudo vim /etc/fail2ban/jail.local
```

Добавьте следующие настройки:
```ini
[nginx-forbidden]
enabled = true
port = http,https
filter = nginx-forbidden
logpath = /var/log/nginx/error.log
maxretry = 3
bantime = 3600
findtime = 600

[sshd]
enabled = true

[nginx-http-auth]
enabled = true

[nginx-botsearch]
enabled = true
```

3. Перезапуск fail2ban:
```bash
sudo systemctl restart fail2ban
```

## Установка приложения

1. Клонирование репозитория:
```bash
sudo git clone https://github.com/olegische/raggen.git /opt/raggen
cd /opt/raggen
```

2. Создание директорий для данных:
```bash
sudo mkdir -p /opt/raggen/data/{faiss,sqlite}
sudo chown -R $USER:$USER /opt/raggen/data
```

3. Настройка переменных окружения:
```bash
# Основной .env файл
cp .env.example .env
# Настройка версии и registry ID
./set-version.sh

# Конфигурация raggen-web
cp raggen-web/.env.example raggen-web/.env
# Настройте API ключи в raggen-web/.env

# Конфигурация raggen-embed
cp raggen-embed/.env.example raggen-embed/.env
```

4. Настройка Nginx:
```bash
# Копирование конфигурации
sudo cp docs/tech/nginx.conf /etc/nginx/sites-available/raggen

# Настройка домена
# Замените actual-domain.com на ваш реальный домен
sudo sed -i 's/your_domain.com/actual-domain.com/' /etc/nginx/sites-available/raggen

# Активация конфигурации
sudo ln -sf /etc/nginx/sites-available/raggen /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

## Запуск приложения

1. Сборка и запуск контейнеров:
```bash
sudo docker-compose build
sudo docker-compose up -d
```

2. Проверка статуса:
```bash
sudo docker-compose ps
sudo docker-compose logs
```

## Обновление приложения

1. Получение обновлений:
```bash
cd /opt/raggen
sudo git pull
```

2. Обновление версии:
```bash
./set-version.sh
```

3. Пересборка и перезапуск:
```bash
sudo docker-compose build
sudo docker-compose up -d
```

## Важные замечания

1. **Монорепозиторий**: Проект состоит из двух сервисов:
   - `raggen-web`: Next.js приложение (порт 3000)
   - `raggen-embed`: FastAPI сервис (порт 8001)

2. **Управление версиями**: 
   - Используйте `set-version.sh` для синхронизации версий между сервисами
   - Версии должны совпадать в `raggen-web/package.json` и `raggen-embed/pyproject.toml`

3. **Данные**:
   - FAISS индексы: `/opt/raggen/data/faiss`
   - SQLite база: `/opt/raggen/raggen-web/prisma/dev.db`

4. **Логи**:
   - Логи контейнеров: `docker-compose logs`
   - Логи Nginx: `/var/log/nginx/raggen.{access,error}.log`

5. **Безопасность**:
   - Все API ключи хранятся в `.env` файлах
   - fail2ban защищает от брутфорс атак
   - Рекомендуется настроить SSL/TLS через certbot

## Устранение неполадок

1. Проверка логов:
```bash
# Логи всех сервисов
sudo docker-compose logs

# Логи отдельных сервисов
sudo docker-compose logs raggen-web
sudo docker-compose logs raggen-embed
```

2. Проверка статуса сервисов:
```bash
sudo docker-compose ps
```

3. Перезапуск сервисов:
```bash
sudo docker-compose restart
```

4. Полная пересборка:
```bash
sudo docker-compose down
sudo docker-compose build --no-cache
sudo docker-compose up -d
``` 