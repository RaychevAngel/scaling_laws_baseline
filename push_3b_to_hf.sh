#!/bin/bash

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Download the model
echo "Downloading Qwen2.5-3B model..."
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-3B .

# Upload to repository
echo "Uploading to repository..."
huggingface-cli upload AngelRaychev/3B-sos-iteration_0 . --repo-type model

# Cleanup
cd -
rm -rf $TEMP_DIR

echo "Done! Repository has been updated with the model." 