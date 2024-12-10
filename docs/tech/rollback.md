# Процедуры отката RAGGEN

## Общие принципы

1. Каждый сервис можно откатить независимо
2. Все образы тегируются версиями
3. Данные каждого сервиса хранятся отдельно:
   - Web: /opt/raggen/data/sqlite
   - Embed: /opt/raggen/data/faiss

## Подготовка к откату

1. Проверка доступных версий:
```bash
# Проверка локальных образов
docker images | grep raggen

# Проверка образов в registry
yc container image list --repository-name raggen-web
yc container image list --repository-name raggen-embed
```

2. Проверка резервных копий данных:
```bash
# Для web сервиса
ls -l /opt/raggen/backups/sqlite/

# Для embed сервиса
ls -l /opt/raggen/backups/faiss/
```

## Откат Web сервиса (VM1)

1. Остановка текущей версии:
```bash
# При использовании docker-compose
docker-compose -f docker-compose.web.yml down

# Или при использовании docker run
docker stop raggen-web
docker rm raggen-web
```

2. Восстановление данных (при необходимости):
```bash
# Создание резервной копии текущих данных
cp -r /opt/raggen/data/sqlite/prisma /opt/raggen/backups/sqlite/prisma_$(date +%Y%m%d_%H%M%S)

# Восстановление из резервной копии
cp -r /opt/raggen/backups/sqlite/prisma_TIMESTAMP /opt/raggen/data/sqlite/prisma
```

3. Запуск предыдущей версии:
```bash
# Обновление версии в .env
VERSION=X.Y.Z ./set-version.sh

# При использовании docker-compose
docker-compose -f docker-compose.web.yml pull
docker-compose -f docker-compose.web.yml up -d

# Или при использовании docker run
docker run -d \
  --name raggen-web \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /opt/raggen/raggen-web/.env:/app/.env \
  -v /opt/raggen/data/sqlite/prisma:/app/prisma \
  cr.yandex/${REGISTRY_ID}/raggen-web:X.Y.Z
```

## Откат Embed сервиса (VM2)

1. Остановка текущей версии:
```bash
# При использовании docker-compose
docker-compose -f docker-compose.embed.yml down

# Или при использовании docker run
docker stop raggen-embed
docker rm raggen-embed
```

2. Восстановление данных (при необходимости):
```bash
# Создание резервной копии текущих данных
cp -r /opt/raggen/data/faiss /opt/raggen/backups/faiss_$(date +%Y%m%d_%H%M%S)

# Восстановление из резервной копии
cp -r /opt/raggen/backups/faiss_TIMESTAMP/* /opt/raggen/data/faiss/
```

3. Запуск предыдущей версии:
```bash
# Обновление версии в .env
VERSION=X.Y.Z ./set-version.sh

# При использовании docker-compose
docker-compose -f docker-compose.embed.yml pull
docker-compose -f docker-compose.embed.yml up -d

# Или при использовании docker run
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
  cr.yandex/${REGISTRY_ID}/raggen-embed:X.Y.Z
```

## Проверка после отката

### Web сервис:
```bash
# Проверка статуса контейнера
docker ps | grep raggen-web
docker logs raggen-web

# Проверка доступности сервиса
curl http://web.your-domain.com
```

### Embed сервис:
```bash
# Проверка статуса контейнера
docker ps | grep raggen-embed
docker logs raggen-embed

# Проверка доступности сервиса
curl http://embed.your-domain.com:8001/docs
```

## Важные замечания

1. **Совместимость данных**:
   - Убедитесь в совместимости схемы БД при откате web сервиса
   - Проверьте совместимость FAISS индексов при откате embed сервиса

2. **Резервное копирование**:
   - Всегда создавайте резервную копию текущих данных перед откатом
   - Храните несколько последних резервных копий
   - Регулярно проверяйте целостность резервных копий

3. **Мониторинг**:
   - Внимательно следите за логами после отката
   - Проверяйте метрики производительности
   - Убедитесь в корректной работе всех функций

4. **Документирование**:
   - Записывайте причины отката
   - Документируйте все проблемы, возникшие при откате
   - Обновляйте процедуры отката на основе полученного опыта
