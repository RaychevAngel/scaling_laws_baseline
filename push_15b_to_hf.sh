#!/bin/bash

# Create a temporary directory
TEMP_DIR=$(mktemp -d)
cd $TEMP_DIR

# Download the model
echo "Downloading Qwen2.5-1.5B model..."
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B .

# Upload to first repository
echo "Uploading to first repository..."
huggingface-cli upload AngelRaychev/1.5B-value-iteration_0 . --repo-type model

# Upload to second repository
echo "Uploading to second repository..."
huggingface-cli upload AngelRaychev/1.5B-policy-iteration_0 . --repo-type model

# Cleanup
cd -
rm -rf $TEMP_DIR

echo "Done! Both repositories have been updated with the model."
