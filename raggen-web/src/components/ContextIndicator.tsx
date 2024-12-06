import { ContextSearchResult } from '@/services/context.service';

interface ContextIndicatorProps {
  context: ContextSearchResult[];
}

export default function ContextIndicator({ context }: ContextIndicatorProps) {
  if (!context || context.length === 0) return null;

  return (
    <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
      <div className="flex items-center gap-1 mb-1">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
        </svg>
        <span>Использованный контекст:</span>
      </div>
      <div className="space-y-1">
        {context.map((ctx, index) => (
          <div key={index} className="flex items-center gap-1">
            <span className="inline-block w-12 text-right">
              {Math.round(ctx.score * 100)}%
            </span>
            <span className="truncate">
              {ctx.message.message}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}