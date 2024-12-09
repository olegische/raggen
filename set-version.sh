#!/bin/bash

# Get version from package.json
WEB_VERSION=$(node -p "require('./raggen-web/package.json').version")

# Get version from pyproject.toml
EMBED_VERSION=$(grep "version = " raggen-embed/pyproject.toml | cut -d'"' -f2)

# Verify versions match
if [ "$WEB_VERSION" != "$EMBED_VERSION" ]; then
    echo "Warning: Versions don't match!"
    echo "raggen-web: $WEB_VERSION"
    echo "raggen-embed: $EMBED_VERSION"
    exit 1
fi

# Export version to .env file
echo "VERSION=$WEB_VERSION" > .env

# Add other environment variables from .env.example
grep -v "^VERSION=" .env.example >> .env

echo "Version $WEB_VERSION has been set in .env" 