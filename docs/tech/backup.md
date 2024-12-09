# Процедуры резервного копирования RAGGEN

## Общие принципы

1. **Периодичность резервного копирования**:
   - Полный бэкап: еженедельно
   - Инкрементальный бэкап: ежедневно
   - Бэкап перед обновлением: обязательно

2. **Компоненты для резервного копирования**:
   - База данных SQLite
   - Индексы FAISS
   - Конфигурационные файлы
   - Логи приложения
   - Docker-образы

## Автоматическое резервное копирование

### 1. Настройка скрипта бэкапа

```bash
#!/bin/bash
# /opt/raggen/scripts/backup.sh

# Настройка переменных
BACKUP_DIR="/backup/raggen"
DATE=$(date +%Y%m%d_%H%M%S)
APP_DIR="/opt/raggen"

# Создание директорий для бэкапа
mkdir -p "${BACKUP_DIR}/{db,faiss,config,logs,docker}"

# Бэкап базы данных
docker-compose exec -T raggen-web cp /app/prisma/dev.db /backup/db/dev.db.${DATE}

# Бэкап индексов FAISS
docker-compose exec -T raggen-embed cp /app/data/faiss/index.faiss /backup/faiss/index.faiss.${DATE}

# Бэкап конфигурации
cp ${APP_DIR}/.env ${BACKUP_DIR}/config/.env.${DATE}
cp ${APP_DIR}/raggen-web/.env ${BACKUP_DIR}/config/raggen-web.env.${DATE}
cp ${APP_DIR}/raggen-embed/.env ${BACKUP_DIR}/config/raggen-embed.env.${DATE}

# Бэкап логов
cp -r ${APP_DIR}/logs/* ${BACKUP_DIR}/logs/

# Сохранение Docker образов
docker save cr.yandex/${REGISTRY_ID}/raggen-web:latest > ${BACKUP_DIR}/docker/raggen-web.${DATE}.tar
docker save cr.yandex/${REGISTRY_ID}/raggen-embed:latest > ${BACKUP_DIR}/docker/raggen-embed.${DATE}.tar

# Очистка старых бэкапов (хранить последние 7 дней)
find ${BACKUP_DIR} -type f -mtime +7 -delete
```

### 2. Настройка Cron

```bash
# Добавление задачи в crontab
0 1 * * * /opt/raggen/scripts/backup.sh >> /var/log/raggen-backup.log 2>&1
```

## Ручное резервное копирование

### 1. Бэкап базы данных

```bash
# Остановка приложения
docker-compose stop raggen-web

# Копирование базы данных
cp raggen-web/prisma/dev.db /backup/raggen/db/dev.db.manual

# Запуск приложения
docker-compose start raggen-web
```

### 2. Бэкап индексов FAISS

```bash
# Остановка сервиса эмбеддингов
docker-compose stop raggen-embed

# Копирование индексов
cp data/faiss/index.faiss /backup/raggen/faiss/index.faiss.manual

# Запуск сервиса
docker-compose start raggen-embed
```

### 3. Бэкап конфигурации

```bash
# Копирование всех конфигурационных файлов
cp .env /backup/raggen/config/
cp raggen-web/.env /backup/raggen/config/raggen-web/
cp raggen-embed/.env /backup/raggen/config/raggen-embed/
```

### 4. Бэкап Docker образов

```bash
# Сохранение текущих образов
docker save cr.yandex/${REGISTRY_ID}/raggen-web:latest > /backup/raggen/docker/raggen-web.tar
docker save cr.yandex/${REGISTRY_ID}/raggen-embed:latest > /backup/raggen/docker/raggen-embed.tar
```

## Проверка резервных копий

### 1. Проверка целостности

```bash
# Проверка базы данных
sqlite3 /backup/raggen/db/dev.db.latest "PRAGMA integrity_check;"

# Проверка архивов Docker
tar tf /backup/raggen/docker/raggen-web.latest.tar
tar tf /backup/raggen/docker/raggen-embed.latest.tar
```

### 2. Тестовое восстановление

Рекомендуется периодически проводить тестовое восстановление в изолированной среде:

```bash
# Создание тестовой среды
mkdir /tmp/raggen-test
cd /tmp/raggen-test

# Копирование бэкапов
cp -r /backup/raggen/* ./

# Тестовое восстановление
docker-compose -f docker-compose.test.yml up -d
```

## Рекомендации по безопасности

1. **Хранение бэкапов**:
   - Использовать шифрование для конфиденциальных данных
   - Хранить копии в разных физических локациях
   - Регулярно проверять доступность бэкапов

2. **Мониторинг**:
   - Настроить оповещения о неудачных бэкапах
   - Следить за размером резервных копий
   - Проверять логи бэкапов

3. **Документация**:
   - Вести журнал резервного копирования
   - Документировать все нестандартные ситуации
   - Обновлять процедуры при изменении системы
``` 