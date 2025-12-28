import os
from pathlib import Path

def create_project_structure():
    """Create the complete project folder structure"""
    
    # Define project root
    project_root = Path("ResNet18-Flower-Classification")
    
    # Define directories to create
    directories = [
        project_root,
        project_root / "data",
        project_root / "src",
        project_root / "configs",
        project_root / "scripts",
        project_root / "checkpoints",
        project_root / "results",
        project_root / "docs" / "report_assets",
    ]
    
    # Create directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Define files to create (with empty content for now)
    files = [
        (project_root / "README.md", "# Project README"),
        (project_root / "requirements.txt", "# Requirements"),
        (project_root / "LICENSE", "# License"),
        (project_root / ".gitignore", "# Git ignore"),
        (project_root / "setup.py", "# Setup script"),
        (project_root / "data" / "generate_dataset.py", "# Dataset generator"),
        (project_root / "data" / "config.yaml", "# Config"),
        (project_root / "src" / "__init__.py", ""),
        (project_root / "src" / "dataset.py", "# Dataset module"),
        (project_root / "src" / "model.py", "# Model module"),
        (project_root / "src" / "train.py", "# Training script"),
        (project_root / "src" / "evaluate.py", "# Evaluation script"),
        (project_root / "src" / "utils.py", "# Utilities"),
        (project_root / "configs" / "default_config.yaml", "# Default config"),
        (project_root / "scripts" / "setup_env.sh", "# Setup env script"),
        (project_root / "scripts" / "setup_env.bat", "# Setup env batch"),
        (project_root / "scripts" / "train.sh", "# Train script"),
        (project_root / "scripts" / "evaluate.sh", "# Evaluate script"),
        (project_root / "checkpoints" / "README.md", "# Checkpoints README"),
        (project_root / "results" / "metrics.json", "# Metrics"),
        (project_root / "docs" / "report_assets" / "README.md", "# Assets README"),
    ]
    
    # Create files
    for file_path, content in files:
        file_path.write_text(content)
        print(f"Created file: {file_path}")
    
    print(f"\nProject structure created at: {project_root.absolute()}")
    print(f"Total directories created: {len(directories)}")
    print(f"Total files created: {len(files)}")

if __name__ == "__main__":
    create_project_structure()