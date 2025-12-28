#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Generate dataset first
echo "Generating synthetic dataset..."
python data/generate_dataset.py \
    --num_classes 3 \
    --samples_per_class 50 \
    --output_dir ./data/synthetic

# Train the model
echo "Starting training..."
python src/train.py \
    --data_dir ./data/synthetic \
    --epochs 5 \
    --batch_size 16 \
    --lr 0.001 \
    --output_dir ./checkpoints

echo "Training completed!"