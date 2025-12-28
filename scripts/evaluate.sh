#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Evaluate the model
echo "Evaluating model..."
python src/evaluate.py \
    --checkpoint ./checkpoints/best_model.pth \
    --data_dir ./data/synthetic \
    --output_dir ./results

echo "Evaluation completed!"