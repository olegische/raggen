# Примеры использования RAGGEN

## Веб-интерфейс (raggen-web)

### Базовое использование чата

1. **Создание нового чата**
```typescript
// Через ChatService
const chatService = new ChatService('yandex');
const response = await chatService.sendMessage('Привет! Как дела?');
console.log(response.message.response);
```

2. **Использование контекста**
```typescript
// С настройками контекста
const response = await chatService.sendMessage('Что мы обсуждали ранее?', chatId, {
  maxContextMessages: 3,
  contextScoreThreshold: 0.7
});
```

3. **Смена провайдера**
```typescript
// Переключение на GigaChat
const gigaChatService = new ChatService('gigachat');
const response = await gigaChatService.sendMessage('Расскажи о себе');
```

### Работа с контекстом

1. **Поиск релевантного контекста**
```typescript
const contextService = new ContextService();
const context = await contextService.searchContext('ключевые слова', {
  maxResults: 5,
  minScore: 0.8
});
```

2. **Форматирование промпта с контекстом**
```typescript
const promptService = new PromptService();
const messages = promptService.formatPromptWithContext(
  'Вопрос пользователя',
  context,
  'yandex',
  {
    maxContextLength: 2000,
    maxContextMessages: 3
  }
);
```

### Настройка параметров генерации

```typescript
const response = await chatService.sendMessage('Сгенерируй креативный текст', chatId, {
  temperature: 0.8,
  maxTokens: 1000,
  model: 'yandexgpt-lite'
});
```

## Сервис эмбеддингов (raggen-embed)

### REST API

1. **Создание эмбеддинга для текста**
```bash
curl -X POST http://localhost:8001/api/v1/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "Пример текста для векторизации"}'
```

Ответ:
```json
{
  "embedding": [...],
  "text": "Пример текста для векторизации",
  "vector_id": 1
}
```

2. **Пакетная обработка текстов**
```bash
curl -X POST http://localhost:8001/api/v1/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Первый текст",
      "Второй текст",
      "Третий текст"
    ]
  }'
```

3. **Поиск похожих текстов**
```bash
curl -X POST http://localhost:8001/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Поисковый запрос",
    "k": 5
  }'
```

### Python SDK

1. **Инициализация клиента**
```python
from raggen_embed import EmbeddingClient

client = EmbeddingClient(base_url="http://localhost:8001")
```

2. **Создание эмбеддинга**
```python
# Один текст
embedding = client.embed_text("Пример текста")
print(f"Vector ID: {embedding.vector_id}")

# Несколько текстов
embeddings = client.embed_texts([
    "Первый текст",
    "Второй текст"
])
```

3. **Поиск похожих**
```python
results = client.search_similar(
    text="Поисковый запрос",
    k=5,
    min_score=0.7
)

for result in results:
    print(f"Score: {result.score}, Text: {result.text}")
```

## Примеры интеграции

### Использование в Next.js компоненте

```typescript
import { useState } from 'react';
import { ChatService } from '@/services/chat.service';
import { ContextSettings } from '@/components/ContextSettings';

export default function ChatComponent() {
  const [messages, setMessages] = useState([]);
  const [contextSettings, setContextSettings] = useState({
    maxContextMessages: 3,
    contextScoreThreshold: 0.7
  });

  const handleSendMessage = async (text) => {
    const chatService = new ChatService('yandex');
    const response = await chatService.sendMessage(text, null, contextSettings);
    
    setMessages(prev => [...prev, {
      text,
      response: response.message.response,
      context: response.context
    }]);
  };

  return (
    <div>
      <ContextSettings
        settings={contextSettings}
        onChange={setContextSettings}
      />
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i}>
            <p>User: {msg.text}</p>
            <p>Bot: {msg.response}</p>
            {msg.context && (
              <div className="context">
                Used context: {msg.context.length} messages
              </div>
            )}
          </div>
        ))}
      </div>
      {/* Input component */}
    </div>
  );
}
```

### Использование в FastAPI приложении

```python
from fastapi import FastAPI, HTTPException
from raggen_embed import EmbeddingClient
from typing import List

app = FastAPI()
embed_client = EmbeddingClient()

@app.post("/process-documents")
async def process_documents(texts: List[str]):
    try:
        # Создаем эмбеддинги для всех текстов
        embeddings = embed_client.embed_texts(texts)
        
        # Ищем похожие для первого текста
        similar = embed_client.search_similar(
            text=texts[0],
            k=3,
            min_score=0.7
        )
        
        return {
            "embeddings_count": len(embeddings),
            "similar_found": len(similar),
            "results": similar
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Рекомендации

1. **Оптимизация контекста**:
   - Используйте разумные значения для `maxContextMessages` (3-5)
   - Настраивайте `contextScoreThreshold` в зависимости от задачи
   - Следите за размером контекста в промпте

2. **Производительность**:
   - Используйте пакетную обработку для множества текстов
   - Кэшируйте часто используемые эмбеддинги
   - Оптими��ируйте размер запросов

3. **Обработка ошибок**:
   - Всегда обрабатывайте ошибки API
   - Используйте retry механизмы
   - Логируйте проблемные случаи

4. **Безопасность**:
   - Валидируйте входные данные
   - Используйте rate limiting
   - Следите за размером запросов 