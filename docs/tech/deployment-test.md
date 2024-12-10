# Сборка и тестирование RAGGEN в тестовой среде

## Подготовка тестовой среды

1. Установка необходимых пакетов:
```bash
sudo apt-get update && sudo apt-get install -y \
    docker.io \
    docker-compose \
    nginx \
    git \
    curl
```

2. Клонирование репозитория:
```bash
git clone https://github.com/olegische/raggen.git
cd raggen
```

3. Настройка переменных окружения:
```bash
cp .env.example .env
# Настройте REGISTRY_ID для вашего registry в Yandex Cloud
```

## Сборка образов для production

1. Сборка всех образов:
```bash
# Установка версии
./set-version.sh

# Сборка образов
docker-compose -f docker-compose.build.yml build
```

2. Проверка собранных образов:
```bash
docker images | grep raggen
```

## Публикация образов в registry

1. Аутентификация в registry:
```bash
# Аутентификация в Yandex Container Registry
yc container registry configure-docker

# Или используя docker login
docker login cr.yandex
```

2. Публикация образов:
```bash
# Push образов в registry
docker-compose -f docker-compose.build.yml push
```

3. Проверка опубликованных образов:
```bash
# Для Yandex Container Registry
yc container image list --repository-name raggen-web
yc container image list --repository-name raggen-embed
```

## Тестирование production конфигурации

1. Создание директорий для данных:
```bash
# Создание структуры как в production
sudo mkdir -p /opt/raggen/data/{faiss,sqlite}
sudo chown -R $USER:$USER /opt/raggen/data
```

2. Тестирование web сервиса:
```bash
# Pull и запуск web сервиса
docker-compose -f docker-compose.web.yml pull
docker-compose -f docker-compose.web.yml up -d

# Проверка статуса
docker-compose -f docker-compose.web.yml ps
docker-compose -f docker-compose.web.yml logs
```

3. Тестирование embed сервиса:
```bash
# Pull и запуск embed сервиса
docker-compose -f docker-compose.embed.yml pull
docker-compose -f docker-compose.embed.yml up -d

# Проверка статуса
docker-compose -f docker-compose.embed.yml ps
docker-compose -f docker-compose.embed.yml logs
```

## Очистка тестовой среды

1. Остановка контейнеров:
```bash
# Остановка всех сервисов
docker-compose -f docker-compose.web.yml down
docker-compose -f docker-compose.embed.yml down
```

2. Очистка данных (опционально):
```bash
# Удаление тестовых данных
sudo rm -rf /opt/raggen/data/faiss/* /opt/raggen/data/sqlite/*
```

3. Очистка образов (опционально):
```bash
# Удаление неиспользуемых образов
docker image prune -a
```

## Важные замечания

1. **Ресурсы**:
   - Убедитесь, что в тестовой среде достаточно CPU и RAM для сборки
   - Для Node.js сборки нужно минимум 2GB RAM
   - Для Python зависимостей нужно минимум 2GB RAM

2. **Registry**:
   - Проверьте права доступа к registry
   - Убедитесь в корректности REGISTRY_ID
   - Проверьте доступность образов после публикации

3. **Версионирование**:
   - Используйте set-version.sh для синхронизации версий
   - Проверяйте версии в тегах образов
   - Убедитесь, что все сервисы используют одинаковую версию

4. **Тестирование**:
   - Проверьте все основные функции
   - Протестируйте взаимодействие между сервисами
   - Убедитесь в корректности настроек CORS
   - Проверьте работу с данными (FAISS, SQLite)

5. **Безопасность**:
   - Не храните чувствительные данные в образах
   - Проверьте настройки CORS
   - Убедитесь в безопасности API endpoints

6. **Документирование**:
   - Записывайте версии успешно собранных образов
   - Документируйте любые специфические настройки
   - Сохраняйте логи сборки для отладки
