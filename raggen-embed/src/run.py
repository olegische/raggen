import uvicorn
from config.settings import Settings

settings = Settings()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,  # Enable auto-reload during development
        log_level=settings.log_level.lower(),
        app_dir="src"
    ) 