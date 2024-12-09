# Настройка окружения RAGGEN

## Системные требования

### Операционная система
- Debian 11 (рекомендуется)
- Другие Linux-дистрибутивы с поддержкой Docker

### Программное обеспечение
- Docker 20.10+
- Docker Compose 2.0+
- Nginx 1.18+
- Git 2.30+
- Python 3.11+
- Node.js 18+

### Аппаратные требования
- CPU: 2+ ядра
- RAM: 4+ ГБ
- Диск: 20+ ГБ

## Подготовка системы

### 1. Установка зависимостей

```bash
# Обновление пакетов
sudo apt-get update

# Установка необходимых пакетов
sudo apt-get install -y \
    docker.io \
    docker-compose \
    nginx \
    git \
    fail2ban \
    vim \
    curl
```

### 2. Настройка безопасности

#### Fail2ban

```bash
# Копирование конфигурации
sudo cp /etc/fail2ban/jail.conf /etc/fail2ban/jail.local

# Настройка правил
sudo tee -a /etc/fail2ban/jail.local << 'EOF'
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
EOF

# Перезапуск сервиса
sudo systemctl restart fail2ban
```

### 3. Настройка директорий

```bash
# Создание рабочей директории
sudo mkdir -p /opt/raggen
sudo chown -R $USER:$USER /opt/raggen

# Создание директорий для данных
sudo mkdir -p /opt/raggen/data/{faiss,sqlite}
sudo chown -R $USER:$USER /opt/raggen/data
```

### 4. Настройка Nginx

```bash
# Создание конфигурации
sudo tee /etc/nginx/sites-available/raggen << 'EOF'
server {
    listen 80;
    server_name your_domain.com;  # Замените на ваш домен

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /api/embed/ {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Логирование
    access_log /var/log/nginx/raggen.access.log;
    error_log /var/log/nginx/raggen.error.log;
}
EOF

# Активация конфигурации
sudo ln -sf /etc/nginx/sites-available/raggen /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl restart nginx
```

## Конфигурация приложения

### 1. Переменные окружения

#### Основной .env файл
```bash
cp .env.example .env
```

Содержимое:
```env
VERSION=1.0.0
REGISTRY_ID=your_registry_id
DATABASE_URL="file:./prisma/dev.db"
NEXT_PUBLIC_EMBED_API_URL=http://raggen-embed:8001
CORS_ORIGINS='["http://localhost:3000"]'
```

#### raggen-web
```bash
cp raggen-web/.env.example raggen-web/.env
```

Необходимые настройки:
- `YANDEX_GPT_API_URL`
- `YANDEX_API_KEY_ID`
- `YANDEX_API_KEY`
- `YANDEX_FOLDER_ID`
- `GIGACHAT_API_URL`
- `GIGACHAT_CREDENTIALS`

#### raggen-embed
```bash
cp raggen-embed/.env.example raggen-embed/.env
```

Основные настройки:
- `HOST=0.0.0.0`
- `PORT=8001`
- `LOG_LEVEL=INFO`
- `FAISS_INDEX_PATH=/app/data/faiss/index.faiss`

### 2. Docker конфигурация

#### Проверка Docker
```bash
# Проверка установки
docker --version
docker-compose --version

# Запуск Docker
sudo systemctl start docker
sudo systemctl enable docker
```

#### Настройка прав
```bash
# Добавление пользователя в группу docker
sudo usermod -aG docker $USER
newgrp docker
```

## Проверка окружения

### 1. Проверка портов
```bash
# Проверка занятости портов
sudo netstat -tulpn | grep -E ':(80|3000|8001)'
```

### 2. Проверка директорий
```bash
# Проверка прав доступа
ls -la /opt/raggen/data
ls -la /opt/raggen/data/faiss
ls -la /opt/raggen/data/sqlite
```

### 3. Проверка логов
```bash
# Проверка логов Nginx
tail -f /var/log/nginx/raggen.error.log
tail -f /var/log/nginx/raggen.access.log

# Проверка логов fail2ban
sudo tail -f /var/log/fail2ban.log
```

## Рекомендации по безопасности

1. **Файрвол**:
   - Разрешить только необходимые порты (80, 443, SSH)
   - Использовать UFW или iptables
   - Настроить rate limiting

2. **SSL/TLS**:
   - Установить certbot
   - Настроить HTTPS
   - Регулярно обновлять сертификаты

3. **Монитор��нг**:
   - Настроить мониторинг системных ресурсов
   - Отслеживать логи на предмет аномалий
   - Настроить алерты

4. **Резервное копирование**:
   - Регулярно бэкапить данные FAISS
   - Бэкапить SQLite базу данных
   - Хранить копии конфигурационных файлов 