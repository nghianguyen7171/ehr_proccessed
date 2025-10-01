#!/bin/bash

# EHR Datasets Environment Setup Script
# This script helps set up the environment for the EHR datasets project

set -e  # Exit on any error

echo "🏥 Setting up EHR Datasets Environment"
echo "======================================"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "✅ Conda found. Setting up conda environment..."
    
    # Create conda environment
    conda env create -f environment.yml --force
    
    echo "✅ Conda environment 'ehr-datasets' created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  conda activate ehr-datasets"
    echo ""
    echo "To install the package in development mode:"
    echo "  pip install -e ."
    
elif command -v python3 &> /dev/null; then
    echo "⚠️  Conda not found. Setting up virtual environment with pip..."
    
    # Create virtual environment
    python3 -m venv ehr-env
    
    # Activate virtual environment
    source ehr-env/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    pip install -r requirements.txt
    
    echo "✅ Virtual environment 'ehr-env' created successfully!"
    echo ""
    echo "To activate the environment, run:"
    echo "  source ehr-env/bin/activate"
    echo ""
    echo "To install the package in development mode:"
    echo "  pip install -e ."
    
else
    echo "❌ Neither conda nor python3 found. Please install Python 3.8+ first."
    exit 1
fi

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate your environment"
echo "2. Run 'pip install -e .' to install the package"
echo "3. Start working with the EHR datasets!"
echo ""
echo "For GPU support, install CUDA-enabled PyTorch:"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"


