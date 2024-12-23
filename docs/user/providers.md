# Провайдеры и модели в RAGGEN

## Yandex GPT

### Особенности
- Поддержка русского и английского языков
- Контекстное окно до 4000 токенов
- Высокая производительность на русскоязычных задачах
- Встроенная фильтрация нежелательного контента

### Возможности
- Генерация текста с учетом контекста
- Ответы на вопросы
- Анализ и обобщение информации
- Работа с кодом и техническими текстами

### Ограничения
- Лимит на количество запросов в минуту
- Фиксированный размер контекстного окна
- Необходимость API ключа Yandex Cloud
- Платный сервис с тарификацией по токенам

## GigaChat

### Особенности
- Специализация на русском языке
- Расширенное контекстное окно до 8000 токенов
- Обучение на российских данных
- Соответствие требованиям безопасности РФ

### Возможности
- Расширенная работа с контекстом
- Глубокое понимание русскоязычного контекста
- Поддержка специфической терминологии
- Работа с длинными документами

### Ограничения
- Необходимость авторизации через Сбер ID
- Региональные ограничения доступа
- Квоты на использование API
- Ограничения на определенные типы контента

## Сравнение моделей

### Размер контекстного окна
| Модель     | Макс. токенов | Оптимальный размер |
|------------|---------------|-------------------|
| Yandex GPT | 4000         | 2000-3000        |
| GigaChat   | 8000         | 4000-6000        |

### Языковые возможности
| Модель     | Русский | Английский | Другие языки |
|------------|---------|------------|--------------|
| Yandex GPT | ✅      | ✅         | ⚠️ Частично |
| GigaChat   | ✅      | ⚠️ Базово  | ❌          |

### Производительность
| Модель     | Скорость ответа | Каче��тво на рус. | Качество на англ. |
|------------|----------------|------------------|------------------|
| Yandex GPT | Высокая        | Отличное         | Хорошее          |
| GigaChat   | Средняя        | Отличное         | Базовое          |

### Стоимость и доступность
| Модель     | Тарификация | Доступность API | Ограничения |
|------------|-------------|-----------------|-------------|
| Yandex GPT | По токенам  | Публичная       | По квотам   |
| GigaChat   | По запросам | По подписке     | Региональные |

## Рекомендации по выбору

### Когда использовать Yandex GPT
- Для задач, требующих работы на русском и английском языках
- При необходимости быстрых ответов
- Для проектов с умеренным объемом контекста
- При важности стабильности API

### Когда использовать GigaChat
- Для работы с большими объемами русскоязычного текста
- При необходимости расширенного контекстного ок��а
- Для проектов с требованиями к локализации данных в РФ
- При работе со специфической русскоязычной терминологией

### Общие рекомендации
1. Оцените требования к размеру контекста
2. Учитывайте языковые требования проекта
3. Проанализируйте ограничения по API и квотам
4. Рассмотрите стоимость использования
5. Проверьте региональную доступность

### Оптимальные сценарии использования

#### Yandex GPT
- Чат-боты с поддержкой нескольких языков
- Генерация технической документации
- Быстрые ответы на вопросы пользователей
- Анализ и обработка коротких текстов

#### GigaChat
- Работа с длинными русскоязычными документами
- Анализ специализированных текстов
- Проекты с требованиями к безопасности данных
- Задачи с глубоким пониманием русского контекста 