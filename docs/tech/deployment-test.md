# Сборка и тестирование RAGGEN

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

## Сборка образов

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

## Тестирование в локальной среде

1. Запуск сервисов для тестирования:
```bash
# Создание необходимых директорий
mkdir -p data/faiss raggen-web/prisma
touch raggen-web/prisma/dev.db

# Запуск с использованием тестового docker-compose
docker-compose up -d
```

2. Проверка работоспособности:
```bash
# Проверка статуса контейнеров
docker-compose ps

# Проверка логов
docker-compose logs

# Проверка доступности сервисов
curl http://localhost:3000/
curl http://localhost:8001/docs
```

## Публикация образов

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

1. Тестирование web сервиса:
```bash
# Pull и запуск web сервиса
docker-compose -f docker-compose.web.yml pull
docker-compose -f docker-compose.web.yml up -d

# Проверка статуса
docker-compose -f docker-compose.web.yml ps
docker-compose -f docker-compose.web.yml logs
```

2. Тестирование embed сервиса:
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
docker-compose down
docker-compose -f docker-compose.web.yml down
docker-compose -f docker-compose.embed.yml down
```

2. Очистка данных (опционально):
```bash
# Удаление тестовых данных
rm -rf data/faiss/* raggen-web/prisma/dev.db
```

3. Очистка образов (опционально):
```bash
# Удаление неиспользуемых образов
docker image prune -a
```

## Важные замечания

1. **Версионирование**:
   - Убедитесь, что версии в package.json и pyproject.toml совпадают
   - Используйте set-version.sh для синхронизации версий
   - Проверяйте версии в тегах образов

2. **Тестирование**:
   - Проверьте все основные функции перед публикацией
   - Протестируйте взаимодействие между сервисами
   - Проверьте работу с разными провайдерами LLM
   - Убедитесь в корректности настроек CORS

3. **Registry**:
   - Проверьте права доступа к registry
   - Убедитесь в корректности REGISTRY_ID
   - Проверьте доступность образов после публикации

4. **Данные**:
   - Проверьте корректность монтирования volumes
   - Убедитесь в правильности прав доступа
   - Проверьте работу с FAISS индексами
   - Протестируйте работу с SQLite базой

5. **Логи**:
   - Проверьте уровни логирования
   - Убедитесь в корректности форматирования логов
   - Проверьте ротацию логов

6. **Безопасность**:
   - Проверьте настройки CORS
   - Убедитесь в безопасности API endpoints
   - Проверьте обработку ошибок
   - Протестируйте механизмы аутентификации
