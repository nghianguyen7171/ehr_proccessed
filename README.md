# ehr-datasets

EHR datasets preprocessing scripts for time-series prediction and tabular data analysis using transformers and deep learning.

The default data split strategy is 70% for training, 10% for validation, and 20% for testing.

## Supported Datasets

- [x] [mimic-iii](https://physionet.org/content/mimiciii/1.4/)
- [x] [mimic-iv](https://www.physionet.org/content/mimiciv/2.2/)
- [x] [eICU](https://physionet.org/content/eicu-crd/2.0/)
- [x] [challenge2012](https://physionet.org/content/challenge-2012/1.0.0/)
- [x] [sepsis](https://physionet.org/content/challenge-2019/1.0.0/)
- [x] [tjh](https://www.nature.com/articles/s42256-020-0180-7) (10-fold)
- [x] [cdsl](https://www.hmhospitales.com/prensa/notas-de-prensa/comunicado-covid-data-save-lives) (10-fold)

## Environment Setup

This project requires Python 3.8+ and focuses on time-series prediction, tabular data processing, and transformer architectures for healthcare data.

### Option 1: Conda Environment (Recommended)

**For CPU-only setup:**
```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate ehr-datasets

# Install the package in development mode
pip install -e .
```

**For GPU setup with CUDA:**
```bash
# Create and activate conda environment with GPU support
conda env create -f environment-gpu.yml
conda activate ehr-datasets-gpu

# Install the package in development mode
pip install -e .
```

**Note:** If you encounter CUDA channel issues, try:
```bash
# Add NVIDIA channel first
conda config --add channels nvidia
conda config --add channels pytorch

# Then create the environment
conda env create -f environment-gpu.yml
```

### Option 2: Pip Virtual Environment

```bash
# Create virtual environment
python -m venv ehr-env
source ehr-env/bin/activate  # On Windows: ehr-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 3: GPU Support (Optional)

For GPU acceleration with CUDA:

```bash
# Install with GPU support
pip install -e .[gpu]

# Or manually install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Development Setup

For development with additional tools:

```bash
# Install with development dependencies
pip install -e .[dev,advanced]

# Set up pre-commit hooks
pre-commit install
```

## Key Dependencies

### Core Libraries
- **Data Processing**: pandas, numpy, scikit-learn
- **Time Series**: statsmodels, tslearn, darts, gluonts
- **Deep Learning**: PyTorch, transformers, pytorch-lightning
- **Tabular ML**: catboost, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn, plotly

### Healthcare-Specific
- **Survival Analysis**: lifelines
- **Imbalanced Data**: imbalanced-learn
- **Medical Statistics**: scipy-stats

## Usage

1. **Data Preprocessing**: Use the preprocessing notebooks in each dataset directory
2. **Time Series Analysis**: Leverage the time-series focused utilities in `utils/tools.py`
3. **Transformer Models**: Utilize the transformer architectures for sequence modeling
4. **Tabular Prediction**: Apply tabular ML methods for structured healthcare data

## Project Structure

```
ehr-datasets/
├── mimic-iii/          # MIMIC-III preprocessing
├── mimic-iv/           # MIMIC-IV preprocessing  
├── eICU/               # eICU preprocessing
├── challenge2012/      # Challenge 2012 preprocessing
├── sepsis/             # Sepsis challenge preprocessing
├── tjh/                # TJH dataset preprocessing
├── cdsl/               # CDSL dataset preprocessing
├── exp/                # Experiments and analysis
├── utils/              # Utility functions
├── requirements.txt    # Python dependencies
├── environment.yml     # Conda environment
└── setup.py           # Package setup
```

## Contributors

This project is brought to you by the following contributors:

- [Yinghao Zhu](https://github.com/yhzhu99)
- [Shiyun Xie](https://github.com/SYXieee)
- [Long He](https://github.com/sh190128)
- [Wenqing Wang](https://github.com/ericaaaaaaaa)
