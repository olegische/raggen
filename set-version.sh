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

# Create temporary file
tmp_file=$(mktemp)

# If .env exists, get REGISTRY_ID from it
if [ -f .env ]; then
    REGISTRY_ID=$(grep "^REGISTRY_ID=" .env | cut -d'=' -f2)
fi

# Copy all lines from .env.example to temp file
cp .env.example "$tmp_file"

# Update VERSION in temp file
sed -i '' "s/^VERSION=.*$/VERSION=$WEB_VERSION/" "$tmp_file" 2>/dev/null || sed -i "s/^VERSION=.*$/VERSION=$WEB_VERSION/" "$tmp_file"

# If we have REGISTRY_ID from existing .env, update it in temp file
if [ ! -z "$REGISTRY_ID" ]; then
    sed -i '' "s/^REGISTRY_ID=.*$/REGISTRY_ID=$REGISTRY_ID/" "$tmp_file" 2>/dev/null || sed -i "s/^REGISTRY_ID=.*$/REGISTRY_ID=$REGISTRY_ID/" "$tmp_file"
fi

# Move temp file to .env
mv "$tmp_file" .env

echo "Version $WEB_VERSION has been set in .env" 