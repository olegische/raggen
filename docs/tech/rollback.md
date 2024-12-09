# Процедуры отката RAGGEN

## Общие принципы

1. **Резервное копирование перед обновлением**:
   - База данных SQLite
   - Индексы FAISS
   - Конфигурационные файлы
   - Логи

2. **Версионирование**:
   - Использование семантического версионирования
   - Сохранение истории обновлений
   - Документирование изменений в CHANGELOG.md

## Откат версии приложения

### 1. Откат через Git

```bash
# Проверка текущей версии
git describe --tags

# Список доступных версий
git tag -l

# Откат к определенной версии
git checkout v1.0.0

# Обновление версии в конфигурации
./set-version.sh

# Пересборка и перезапуск
docker-compose down
docker-compose build
docker-compose up -d
```

### 2. Откат через Docker

```bash
# Список образов
docker images

# Откат к предыдущему образу
docker-compose down
docker tag cr.yandex/${REGISTRY_ID}/raggen-web:previous cr.yandex/${REGISTRY_ID}/raggen-web:latest
docker tag cr.yandex/${REGISTRY_ID}/raggen-embed:previous cr.yandex/${REGISTRY_ID}/raggen-embed:latest
docker-compose up -d
```

## Откат базы данных

### 1. Восстановление SQLite

```bash
# Остановка сервисов
docker-compose down

# Восстановление из бэкапа
cp /backup/raggen/db/dev.db.backup raggen-web/prisma/dev.db

# Запуск сервисов
docker-compose up -d
```

### 2. Откат миграций Prisma

```bash
# Просмотр истории миграций
npx prisma migrate status

# Откат к определенной миграции
npx prisma migrate reset --to <migration_name>

# Применение миграции
npx prisma migrate deploy
```

## Откат векторного хранилища

### 1. Восстановление индексов FAISS

```bash
# Остановка сервиса эмбеддингов
docker-compose stop raggen-embed

# Восстановление из бэкапа
cp /backup/raggen/faiss/index.faiss.backup data/faiss/index.faiss

# Запуск сервиса
docker-compose start raggen-embed
```

### 2. Пересоздание индексов

```bash
# В случае повреждения индекса
rm -f data/faiss/index.faiss

# Сервис автоматически создаст новый индекс при запуске
docker-compose restart raggen-embed
```

## Откат конфигурации

### 1. Конфигурация приложения

```bash
# Восстановление .env файлов
cp /backup/raggen/.env.backup .env
cp /backup/raggen/raggen-web/.env.backup raggen-web/.env
cp /backup/raggen/raggen-embed/.env.backup raggen-embed/.env

# Перезапуск с новой конфигурацией
docker-compose down
docker-compose up -d
```

### 2. Конфигурация Nginx

```bash
# Восстановление конфигурации
sudo cp /backup/raggen/nginx/raggen.conf /etc/nginx/sites-available/raggen

# Проверка конфигурации
sudo nginx -t

# Применение изменений
sudo systemctl restart nginx
```

## Проверка после отката

### 1. Проверка сервисов

```bash
# Статус контейнеров
docker-compose ps

# Проверка логов
docker-compose logs

# Проверка доступности API
curl http://localhost:3000/health
curl http://localhost:8001/health
```

### 2. Проверка данных

```bash
# Проверка базы данных
npx prisma studio

# Проверка индексов FAISS
curl -X POST http://localhost:8001/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"text": "test", "k": 1}'
```

## План отката для критических ситуаций

1. **Полный откат системы**:
   ```bash
   # Остановка всех сервисов
   docker-compose down
   
   # Восстановление всех данных
   ./scripts/restore-backup.sh <backup_date>
   
   # Запуск предыдущей версии
   git checkout <previous_version>
   docker-compose up -d
   ```

2. **Быстрый откат с потерей данных**:
   ```bash
   # Только в крайнем случае
   docker-compose down -v
   git checkout <stable_version>
   ./scripts/clean-install.sh
   ```

## Рекомендации

1. **Перед обновлением**:
   - Создать полный бэкап системы
   - Проверить наличие места для отката
   - Документировать текущую версию

2. **Во время отката**:
   - Следовать чек-листу отката
   - Проверять каж��ый шаг
   - Вести лог действий

3. **После отката**:
   - Проверить работоспособность
   - Проанализировать причины отката
   - Обновить документацию

4. **Мониторинг**:
   - Следить за логами
   - Проверять метрики
   - Тестировать основной функционал 