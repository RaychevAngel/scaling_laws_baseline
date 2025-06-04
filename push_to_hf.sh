#!/bin/bash

# Create temporary directory
TEMP_DIR="temp_model"
mkdir -p $TEMP_DIR
cd $TEMP_DIR

# Download the model
echo "Downloading Qwen2.5-1.5B model..."
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B .

# Function to create or update repository
create_or_update_repo() {
    local repo_name=$1
    echo "Setting up repository: $repo_name"
    
    # Try to create the repository
    if ! huggingface-cli repo create $repo_name 2>/dev/null; then
        echo "Repository $repo_name already exists, proceeding with update..."
    fi
    
    # Initialize git and push
    git init
    git add .
    git commit -m "Initial commit with Qwen2.5-1.5B model"
    git remote add origin https://huggingface.co/$repo_name || git remote set-url origin https://huggingface.co/$repo_name
    git push -u origin main --force
}

# Create/Update first repository
create_or_update_repo "AngelRaychev/1.5B-value-iteration_0"

# Create/Update second repository
cd ..
mkdir -p temp_model_2
cd temp_model_2
cp -r ../$TEMP_DIR/* .
create_or_update_repo "AngelRaychev/1.5B-policy-iteration_0"

# Cleanup
cd ..
rm -rf $TEMP_DIR temp_model_2

echo "Done! Both repositories have been created/updated with the model."
