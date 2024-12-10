#!/bin/bash

# Get version from package.json using grep
WEB_VERSION=$(grep '"version":' raggen-web/package.json | cut -d'"' -f4)

# Get version from pyproject.toml
EMBED_VERSION=$(grep "version = " raggen-embed/pyproject.toml | cut -d'"' -f2)

# Verify versions match
if [ "$WEB_VERSION" != "$EMBED_VERSION" ]; then
    echo "Warning: Versions don't match!"
    echo "raggen-web: $WEB_VERSION"
    echo "raggen-embed: $EMBED_VERSION"
    exit 1
fi

# Create temporary file
tmp_file=$(mktemp)

# If .env exists, get REGISTRY_ID from it
if [ -f .env ]; then
    REGISTRY_ID=$(grep "^REGISTRY_ID=" .env | cut -d'=' -f2)
fi

# Copy all lines from .env.example to temp file
cp .env.example "$tmp_file"

# Add VERSION to the beginning of the file
echo "VERSION=$WEB_VERSION" > .env
cat "$tmp_file" >> .env

# If we have REGISTRY_ID from existing .env, update it
if [ ! -z "$REGISTRY_ID" ]; then
    sed -i '' "s/^REGISTRY_ID=.*$/REGISTRY_ID=$REGISTRY_ID/" .env 2>/dev/null || sed -i "s/^REGISTRY_ID=.*$/REGISTRY_ID=$REGISTRY_ID/" .env
fi

# Clean up
rm "$tmp_file"

echo "Version $WEB_VERSION has been set in .env"
