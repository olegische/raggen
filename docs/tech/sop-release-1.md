# SOP: Release 1.0 - Интеграция RAG

## 1. Подготовка инфраструктуры

### 1.1. Настройка монорепозитория
- [x] Создать базовую структуру монорепозитория
- [x] Настроить .gitignore для обоих сервисов
- [x] Настроить линтеры и форматтеры
- [x] Создать базовые README файлы

### 1.2. Настройка raggen-web
- [x] Инициализировать Next.js проект
- [x] Настроить TypeScript
- [x] Настроить Tailwind CSS
- [x] Настроить Prisma
- [x] Создать базовую структуру директорий

### 1.3. Настройка raggen-embed
- [x] Создать Python проект с FastAPI
- [x] Настроить виртуальное окружение
- [x] Установить sentence-transformers и FAISS
- [x] Создать базовую структуру директорий
- [x] Настроить конфигурацию проекта

## 2. Разработка raggen-embed

### 2.1. Базовая структура
- [x] Создать основной FastAPI application
- [x] Настроить CORS для взаимодействия с raggen-web
- [x] Настроить логирование
  - [x] Реализовать структурированное логирование
  - [x] Добавить request ID для отслеживания запросов
  - [x] Настроить форматирование логов
  - [x] Добавить уровни логирования
- [x] Добавить конфигурацию через environment variables

### 2.2. Модель эмбеддингов
- [x] Интегрировать модель all-MiniLM-L6-v2
- [x] Создать сервис для работы с моделью
- [x] Реализовать кэширование модели
- [x] Добавить обработку ошибок модели
- [x] Написать тесты для модели эмбеддингов
  - [x] Тесты инициализации модели
  - [x] Тесты генерации эмбеддингов
  - [x] Тесты обработки ошибок
  - [x] Тесты кэширования

### 2.3. Векторное хранилище
- [x] Настроить FAISS для 384-мерных векторов
- [x] Реализовать сохранение векторов
- [x] Реализовать поиск похожих векторов
- [x] Добавить персистентность индекса FAISS
- [x] Написать тесты для векторного хранилища
  - [x] Тесты инициализации FAISS
  - [x] Тесты добавления векторов
  - [x] Тесты поиска векторов
  - [x] Тесты производительности

### 2.4. API Endpoints
- [x] Создать endpoint для векторизации текста
- [x] Создать endpoint для поиска похожих текстов
- [x] Добавить валидацию входных данных
- [x] Реализовать обработку ошибок
- [x] Написать тесты для API
  - [x] Тесты эндпоинтов
  - [x] Тесты валидации
  - [x] Тесты обработки ошибок
  - [x] Интеграционные тесты

### 2.5. Тестирование
- [x] Написать unit тесты для сервисов
  - [x] Тесты эмбеддингов
  - [x] Тесты векторного хранилища
  - [x] Тесты конфигурации
- [x] Написать интеграционные тесты для API
  - [x] Тесты основного приложения
  - [x] Тесты CORS
  - [x] Тесты middleware
- [x] Добавить тесты производительности
  - [x] Тесты времени отклика
  - [x] Тесты под нагрузкой
- [ ] Настроить CI для тестов

## 3. Обновление raggen-web

### 3.1. База данных
- [x] Обновить схему Prisma для хранения контекста
- [x] Создать миграции
- [x] Добавить индексы для оптимизации
- [x] Реализовать сервисный слой для работы с БД
- [x] Написать тесты для сервисного слоя

### 3.2. Интеграция с raggen-embed
- [x] Создать клиент для raggen-embed API
- [x] Реализовать сервис для работы с эмбеддингами
- [x] Добавить обработку ошибок
- [x] Реализовать retry механизм
- [x] Написать тесты для клиента и сервиса
- [x] Протестировать обработку ошибок и retry логику

### 3.3. Обработка контекста
- [x] Реализовать сервис поиска контекста
- [x] Создать форматтер промптов с контекстом
- [x] Добавить приоритизацию контекста
- [x] Реализовать кэширование контекста
- [x] Написать тесты для поиска контекста
- [x] Протестировать форматирование промптов
- [x] Проверить эффективность кэширования

### 3.4. UI компоненты
- [x] Добавить индикацию использованного контекста
- [x] Обновить компонент чата
- [x] Добавить настройки контекста
- [x] Реализовать отображение истории с контекстом
- [x] Написать тесты для UI компонентов
- [x] Протестировать интерактивность
- [x] Проверить доступность (a11y)

### 3.5. Тестирование
- [x] Написать unit тесты для новых сервисов
- [x] Обновить тесты компонентов
- [x] Добавить интеграционные тесты
- [x] Протестировать производительность

## 4. Интеграция и тестирование

### 4.1. Локальное тестирование ⚠️
- [x] Настроить локальное кружение для обоих сервисов
- [x] Протестировать взаимодействие сервисов
- [x] Проверить обработку ошибок
- [ ] Замерить производительность

### 4.2. Нагрузочное тестирование ❌
- [ ] Протестировать векторизацию под нагрузкой
- [ ] Проверить производительность поиска
- [ ] Оценить потребление памяти FAISS
- [ ] Оптимизировать узкие места

### 4.3. Интеграционное тестирование ⚠️
- [x] Проверить работу с разными провайдерами
- [x] Протестировать различные сценарии использования
- [ ] Проверить обработку edge cases
- [ ] Валидировать качество контекста

## 5. Документация

### 5.1. Техническая документация ✅
- [x] Описать API endpoints
- [x] Документировать схему базы данных
- [x] Описать конфигурацию сервисов
- [x] Добавить примеры использования

### 5.2. Документация по равертыванию ✅
- [x] Создать инструкции по установке
- [x] Описать конфигурацию окружения
- [x] Добавить примеры конфигурационных файлов
- [x] Описать процедуры отката

### 5.3. Пользовательская документация ⚠️
- [x] Создать структуру пользовательской документации
  - [x] Разделить техническую и пользовательскую документацию
  - [x] Создать оглавление и основные разделы
  - [x] Обновить ссылки в основном README
- [x] Создать руководство по началу работы
  - [x] Описать системные требования
  - [x] Добавить инструкции по установке
  - [x] Описать первый запуск
  - [x] Описать базовую настройку
- [x] Создать документацию по основным функциям
  - [x] Описать работу с чатом
  - [x] Описать выбор моделей
  - [x] Описать настройку контекста
  - [x] Описать управление историей
- [x] Создать руководство по настройке
  - [x] Описать параметры генерации
  - [x] Описать настройки контекста
  - [x] Описать темы оформления
  - [x] Описать горячие клавиши
- [x] Создать документацию по работе с контекстом
  - [x] Описать принципы работы контекста
  - [x] Объяснить контекстный поиск
  - [x] Описать настройки релевантности
  - [x] Добавить примеры использования
- [x] Создать документацию по провайдерам
  - [x] Описать особенности Yandex GPT
  - [x] Описать особенности GigaChat
  - [x] Добавить сравнение моделей
  - [x] Дать рекомендации по выбору
- [x] Создать FAQ
  - [x] Собрать общие вопросы
  - [x] Описать типичные проблемы и решения
  - [x] Добавить советы и рекомендации
- [x] Создать глоссарий
  - [x] Определить основные термины
  - [x] Добавить сокращения
  - [x] Дополнить техническими терминами

## 6. Развертывание ⚠️

### 6.1. Подготовка ⚠️
- [x] Проверить все зависимости
- [x] Подготовить конфигурационные файлы
- [ ] Проверить доступы к API провайдеров
- [ ] Подготовить скрипты развертывания
- [x] Настроить процедуры резервного копирования
  - [x] Настроить автоматические бэкапы
  - [x] Проверить скрипты восстановления
  - [x] Настроить ротацию бэкапов
  - [x] Проверить доступность бэкапов

### 6.2. Развертывание сервисов ⚠️
- [x] Развернуть raggen-embed (локально)
- [x] Развернуть raggen-web (локально)
- [x] Настроить взаимодейтв между сервисами
- [ ] Проверить логирование

### 6.3. Пост-развертывание ❌
- [ ] Проверить все функции в production
- [ ] Мониторить производительность
- [ ] Проверить обработку ошибок
- [ ] Валидировать качество ответов
- [ ] Проверить выполнение резервного копирования
  - [ ] Подтвердить создание бэкапов по расписанию
  - [ ] Проверить целостность бэкапов
  - [ ] Протестировать процедуру восстановления
  - [ ] Проверить логи бэкапов

## 7. Критерии приемки

### 7.1. Функциональные требования
- [ ] Векторизация сообщений работает корректно
- [ ] Поиск контекста возвращает релевантные результаты
- [ ] Промпты формируются с учетом контекста
- [ ] UI корректно отображает использованный контекст

### 7.2. Нефункциональные требования
- [ ] Время ответа API не превышает 500мс
- [ ] Потребление памяти FAISS в пределах нормы
- [ ] Все тесты проходят успешно
- [ ] Документация полная и актуальная