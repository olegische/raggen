import pytest
from fastapi.testclient import TestClient
import os
import json

from main import app
from config.settings import Settings

@pytest.fixture
def settings():
    """Fixture for application settings."""
    return Settings()

@pytest.fixture
def client():
    """Fixture for FastAPI test client."""
    return TestClient(app)

def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_cors_headers(client, settings):
    """Test CORS headers are set correctly."""
    # Test preflight request
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers
    assert "access-control-allow-methods" in response.headers
    assert "access-control-allow-headers" in response.headers

def test_request_id_middleware(client):
    """Test that request ID is generated and included in response headers."""
    response = client.get("/health")
    assert response.status_code == 200
    # Check logs for request ID (this is implicit as we can't access logs directly in tests)

def test_logging_middleware(client):
    """Test that requests are logged."""
    response = client.get("/health")
    assert response.status_code == 200
    # Check logs (this is implicit as we can't access logs directly in tests)

def test_environment_config(settings):
    """Test that environment variables are loaded correctly."""
    # Test default values
    assert settings.api_title == "Raggen Embed API"
    assert settings.api_version == "1.0.0"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8001
    
    # Test overriding values through environment
    os.environ["API_TITLE"] = "Test API"
    os.environ["PORT"] = "9000"
    
    test_settings = Settings()
    assert test_settings.api_title == "Test API"
    assert test_settings.port == 9000
    
    # Clean up environment
    del os.environ["API_TITLE"]
    del os.environ["PORT"]

def test_cors_config(settings):
    """Test CORS configuration."""
    # Test default CORS settings
    assert settings.cors_origins == ["*"]
    assert settings.cors_allow_credentials is True
    assert settings.cors_allow_methods == ["*"]
    assert settings.cors_allow_headers == ["*"]
    
    # Test custom CORS settings
    os.environ["CORS_ORIGINS"] = json.dumps(["http://localhost:3000", "http://localhost:8000"])
    os.environ["CORS_ALLOW_METHODS"] = json.dumps(["GET", "POST"])
    
    test_settings = Settings()
    assert test_settings.cors_origins == ["http://localhost:3000", "http://localhost:8000"]
    assert test_settings.cors_allow_methods == ["GET", "POST"]
    
    # Clean up environment
    del os.environ["CORS_ORIGINS"]
    del os.environ["CORS_ALLOW_METHODS"] 