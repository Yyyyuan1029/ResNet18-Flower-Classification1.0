#!/bin/bash

echo "Setting up ResNet18 Flower Classification Environment..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/synthetic
mkdir -p checkpoints
mkdir -p results
mkdir -p logs

echo "Environment setup complete!"
echo "To activate virtual environment: source venv/bin/activate"