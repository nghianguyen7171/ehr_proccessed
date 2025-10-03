# VitalDB Clinical Data Processing

This module provides comprehensive functions to process VitalDB clinical data from raw CSV files to integrated clinical datasets and time-series sequences suitable for deep learning models.

## 📁 Directory Structure

```
vitaldb/
├── vitaldb_clinical_process.py    # Main processing module
├── README.md                      # This documentation
└── processed/                     # Output directory (created during processing)
    ├── integrated_clinical_data.csv
    ├── time_series_sequences.npz
    └── lab_parameter_mapping.json
```

## 🚀 Quick Start

### Basic Usage

```python
from vitaldb.vitaldb_clinical_process import VitalDBProcessor

# Initialize processor
processor = VitalDBProcessor("/path/to/vitaldb/csv/files")

# Complete processing pipeline
csv_data = processor.load_csv_files()
relationships = processor.explore_dataset_relationships()
param_mapping = processor.create_lab_parameter_mapping()
integrated_data = processor.create_integrated_clinical_dataset()
sequences, targets, patient_info = processor.create_time_series_sequences()

# Save processed data
processor.save_integrated_data("/path/to/output/directory")
```

### Command Line Usage

```bash
python vitaldb_clinical_process.py
```

## 📊 Input Data Requirements

The processor expects the following CSV files in the input directory:

### Required Files

1. **`clinical_data.csv`** - Patient demographics and clinical information
   - Must contain: `caseid`, `subjectid`, `age`, `sex`, `death_inhosp`
   - Additional columns: `height`, `weight`, `bmi`, `asa`, etc.

2. **`lab_data.csv`** - Laboratory test results and measurements
   - Must contain: `caseid`, `dt` (time), `name` (parameter), `result` (value)
   - Time-series format with multiple parameters per patient

3. **`lab_parameters.csv`** - Parameter definitions and metadata
   - Must contain: `Parameter`, `Description`, `Unit`, `Category`
   - Used for parameter interpretation and mapping

### Optional Files

4. **`clinical_parameters.csv`** - Clinical parameter definitions
5. **`track_names.csv`** - Vital sign track identifiers

## 🔄 Processing Pipeline

### Step 1: Data Loading and Analysis

```python
# Load all CSV files
csv_data = processor.load_csv_files()

# Analyze each dataset
for file_name, df in csv_data.items():
    analysis = processor.analyze_csv_file(df, file_name)
    print(f"Dataset: {file_name}")
    print(f"  Shape: {analysis['shape']}")
    print(f"  Completeness: {analysis['completeness']:.1f}%")
    print(f"  Missing values: {sum(analysis['missing_summary'].values())}")
```

**What happens:**
- Loads all CSV files from the specified directory
- Performs comprehensive analysis of each dataset
- Calculates data quality metrics (completeness, missing values, duplicates)
- Identifies data types and column characteristics

### Step 2: Dataset Relationship Exploration

```python
# Explore relationships between datasets
relationships = processor.explore_dataset_relationships()

print(f"Case ID overlap: {relationships['caseid_overlap']['coverage']:.1f}%")
print(f"Lab parameter coverage: {relationships['lab_parameter_overlap']['coverage']:.1f}%")
```

**What happens:**
- Identifies common patients between clinical and lab data
- Maps lab parameters to their definitions
- Calculates coverage statistics
- Validates data consistency across files

### Step 3: Parameter Mapping Creation

```python
# Create lab parameter mapping
param_mapping = processor.create_lab_parameter_mapping()

# Example usage
for param, info in param_mapping.items():
    print(f"{param}: {info['Description']} ({info['Unit']})")
```

**What happens:**
- Creates mapping from parameter names to descriptions and units
- Enables clinical interpretation of lab values
- Provides context for feature engineering

### Step 4: Integrated Clinical Dataset Creation

```python
# Create integrated clinical dataset
integrated_data = processor.create_integrated_clinical_dataset()

print(f"Integrated dataset shape: {integrated_data.shape}")
print(f"Features added: {integrated_data.shape[1] - original_clinical.shape[1]}")
```

**What happens:**
- Merges clinical data with lab data using patient IDs (`caseid`)
- Creates summary statistics for each patient's lab data:
  - `dt_count`: Number of lab measurements
  - `result_mean/std/min/max`: Statistical summaries
  - Parameter-specific summaries (e.g., `wbc_mean`, `hb_std`)

**Output columns:**
- Original clinical features (age, sex, height, weight, etc.)
- Lab summary statistics (count, mean, std, min, max)
- Parameter-specific summaries for key lab values

### Step 5: Time-Series Sequence Creation

```python
# Create time-series sequences for deep learning
sequences, targets, patient_info = processor.create_time_series_sequences(
    max_length=48,        # Sequence length
    min_lab_records=10,   # Minimum records per patient
    max_patients=100      # Limit for processing
)

print(f"Created {len(sequences)} sequences")
print(f"Sequence shape: {sequences[0].shape}")
print(f"Target distribution: {pd.Series(targets).value_counts().to_dict()}")
```

**What happens:**
- Selects patients with sufficient lab data (≥10 records)
- Creates fixed-length sequences (e.g., 48 time points)
- Generates overlapping sequences for more training data
- Handles missing values with forward-fill and backward-fill
- Pads sequences to consistent length
- Associates each sequence with clinical outcomes

**Sequence structure:**
- **Input**: `(sequence_length, num_parameters)` - e.g., (48, 33)
- **Target**: Binary outcome (0/1) for mortality prediction
- **Metadata**: Patient ID, sequence length, time range

### Step 6: Data Saving

```python
# Save processed data
processor.save_integrated_data("/path/to/output")

# Files created:
# - integrated_clinical_data.csv
# - time_series_sequences.npz
# - lab_parameter_mapping.json
```

## 📈 Advanced Usage

### Custom Sequence Creation

```python
# Custom parameters for sequence creation
sequences, targets, patient_info = processor.create_time_series_sequences(
    max_length=96,        # Longer sequences
    min_lab_records=20,   # More strict filtering
    max_patients=None     # Process all patients
)
```

### Loading Previously Processed Data

```python
# Load previously processed data
processor = VitalDBProcessor("dummy_path")  # Path not used for loading
processor.load_processed_data("/path/to/processed/data")

# Access loaded data
integrated_data = processor.integrated_clinical
sequences = processor.sequences
targets = processor.targets
```

### Data Quality Assessment

```python
# Get comprehensive data summary
summary = processor.get_data_summary()

print(f"CSV files: {summary['csv_files']}")
print(f"Integrated dataset: {summary['integrated_clinical_shape']}")
print(f"Sequences: {summary['sequences_count']}")
print(f"Target distribution: {summary['target_distribution']}")
```

## 🔧 Configuration Options

### VitalDBProcessor Parameters

```python
processor = VitalDBProcessor(
    data_dir="/path/to/vitaldb/csv/files"  # Required: CSV files directory
)
```

### create_time_series_sequences Parameters

```python
sequences, targets, patient_info = processor.create_time_series_sequences(
    max_length=48,        # Maximum sequence length (default: 48)
    min_lab_records=10,   # Minimum lab records per patient (default: 10)
    max_patients=None     # Maximum patients to process (default: None)
)
```

## 📊 Output Data Formats

### 1. Integrated Clinical Data (`integrated_clinical_data.csv`)

**Structure:**
```csv
caseid,age,sex,height,weight,death_inhosp,dt_count,result_mean,wbc_mean,hb_mean,...
123,65,M,175,80,0,45,12.5,8.2,14.3,...
456,72,F,160,65,1,38,15.2,7.8,12.1,...
```

**Columns:**
- **Patient identifiers**: `caseid`, `subjectid`
- **Demographics**: `age`, `sex`, `height`, `weight`, `bmi`
- **Outcomes**: `death_inhosp`, `los` (if available)
- **Lab summaries**: `dt_count`, `result_mean/std/min/max`
- **Parameter-specific**: `wbc_mean`, `hb_std`, `na_min`, etc.

### 2. Time-Series Sequences (`time_series_sequences.npz`)

**Structure:**
```python
# Load sequences
data = np.load('time_series_sequences.npz')
sequences = data['sequences']      # Shape: (N, 48, 33)
targets = data['targets']          # Shape: (N,)
patient_info = data['patient_info'] # List of dicts
```

**Content:**
- **sequences**: 3D numpy array (N sequences × 48 time points × 33 parameters)
- **targets**: 1D numpy array (binary outcomes for each sequence)
- **patient_info**: List of dictionaries with metadata

### 3. Parameter Mapping (`lab_parameter_mapping.json`)

**Structure:**
```json
{
  "wbc": {
    "Description": "White Blood Cell Count",
    "Unit": "10^9/L",
    "Category": "Hematology"
  },
  "hb": {
    "Description": "Hemoglobin",
    "Unit": "g/dL",
    "Category": "Hematology"
  }
}
```

## 🏥 Clinical Interpretation

### Lab Parameter Categories

The processor identifies and categorizes lab parameters:

- **Hematology**: `wbc`, `hb`, `hct`, `plt` (blood cell counts)
- **Chemistry**: `na`, `k`, `gluc`, `alb`, `cr`, `bun` (electrolytes, kidney function)
- **Coagulation**: `pt`, `ptt`, `inr` (clotting parameters)
- **Arterial Blood Gas**: `ph`, `pco2`, `po2` (acid-base balance)

### Time-Series Interpretation

Each sequence represents:
- **48 time points** of lab measurements for one patient
- **Multiple parameters** measured simultaneously
- **Temporal patterns** that can indicate clinical deterioration
- **Missing value patterns** that may be clinically significant

### Missing Value Handling

The processor handles missing values using:
1. **Forward Fill**: Use last known value (clinically reasonable for continuous monitoring)
2. **Backward Fill**: Use next known value (for initial missing values)
3. **Padding**: Repeat last values for sequences shorter than required length

## 🤖 Deep Learning Integration

### Transformer Model Input

```python
# Prepare data for transformer training
X = np.array(sequences)  # Shape: (N, 48, 33)
y = np.array(targets)    # Shape: (N,)

# For PyTorch
import torch
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# For TensorFlow/Keras
import tensorflow as tf
X_tensor = tf.constant(X, dtype=tf.float32)
y_tensor = tf.constant(y, dtype=tf.int32)
```

### Model Architecture Example

```python
# Simple transformer for mortality prediction
import torch.nn as nn

class VitalDBTransformer(nn.Module):
    def __init__(self, input_dim=33, seq_length=48, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, nhead=8),
            num_layers=4
        )
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)
```

## 🔍 Troubleshooting

### Common Issues

1. **FileNotFoundError**: Check that CSV files exist in the specified directory
2. **MemoryError**: Reduce `max_patients` parameter or process in batches
3. **Empty sequences**: Increase `min_lab_records` or check data quality
4. **Missing parameters**: Verify `lab_parameters.csv` contains all required columns

### Data Quality Issues

```python
# Check data quality before processing
for file_name, df in csv_data.items():
    analysis = processor.analyze_csv_file(df, file_name)
    if analysis['completeness'] < 50:
        print(f"Warning: {file_name} has low completeness ({analysis['completeness']:.1f}%)")
```

### Performance Optimization

```python
# For large datasets, process in batches
batch_size = 1000
for i in range(0, len(eligible_patients), batch_size):
    batch_patients = eligible_patients[i:i+batch_size]
    # Process batch...
```

## 📚 Examples

### Complete Processing Example

```python
#!/usr/bin/env python3

from vitaldb.vitaldb_clinical_process import VitalDBProcessor
import pandas as pd
import numpy as np

def main():
    # Configuration
    data_dir = "/path/to/vitaldb/csv/files"
    output_dir = "/path/to/processed/output"
    
    # Initialize processor
    processor = VitalDBProcessor(data_dir)
    
    # Complete pipeline
    print("Loading CSV files...")
    csv_data = processor.load_csv_files()
    
    print("Creating integrated dataset...")
    integrated_data = processor.create_integrated_clinical_dataset()
    
    print("Creating time-series sequences...")
    sequences, targets, patient_info = processor.create_time_series_sequences(
        max_length=48,
        min_lab_records=10
    )
    
    print("Saving processed data...")
    processor.save_integrated_data(output_dir)
    
    # Analysis
    summary = processor.get_data_summary()
    print(f"\nProcessing complete!")
    print(f"Integrated dataset: {summary['integrated_clinical_shape']}")
    print(f"Sequences created: {summary['sequences_count']}")
    print(f"Target distribution: {summary['target_distribution']}")

if __name__ == "__main__":
    main()
```

### Custom Analysis Example

```python
# Custom analysis of integrated data
integrated_data = processor.integrated_clinical

# Analyze mortality by lab parameters
mortality_analysis = integrated_data.groupby('death_inhosp').agg({
    'wbc_mean': ['mean', 'std'],
    'hb_mean': ['mean', 'std'],
    'age': ['mean', 'std']
}).round(2)

print("Mortality analysis:")
print(mortality_analysis)

# Time-series analysis
sequences = processor.sequences
targets = processor.targets

# Calculate sequence statistics
seq_lengths = [len(seq) for seq in sequences]
print(f"\nSequence statistics:")
print(f"Average length: {np.mean(seq_lengths):.1f}")
print(f"Min length: {min(seq_lengths)}")
print(f"Max length: {max(seq_lengths)}")
```

## 📖 References

- **VitalDB Dataset**: [https://vitaldb.net](https://vitaldb.net)
- **Time-Series Deep Learning**: [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
- **Clinical Data Processing**: [MIMIC-III Processing Guide](https://mimic.physionet.org/)

## 🤝 Contributing

To contribute to this module:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This module is part of the EHR Datasets project and follows the same license terms.

---

**For questions or issues, please refer to the main project documentation or create an issue in the repository.**
