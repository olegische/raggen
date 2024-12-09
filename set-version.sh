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

# If .env exists, copy all lines except VERSION to temp file
if [ -f .env ]; then
    grep -v "^VERSION=" .env > "$tmp_file"
else
    # If .env doesn't exist, copy from .env.example
    cp .env.example "$tmp_file"
fi

# Add new VERSION to the beginning of the file
echo "VERSION=$WEB_VERSION" > .env
cat "$tmp_file" >> .env

# Clean up
rm "$tmp_file"

echo "Version $WEB_VERSION has been set in .env" 