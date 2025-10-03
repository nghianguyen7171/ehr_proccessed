# VitalDB Clinical Data Processing Steps

This document provides a detailed step-by-step explanation of how the VitalDB clinical data is processed from raw CSV files to integrated datasets and time-series sequences suitable for deep learning models.

## 📋 Overview

The VitalDB processing pipeline transforms raw clinical data into two main outputs:
1. **Integrated Clinical Dataset**: Merged clinical and lab data with summary statistics
2. **Time-Series Sequences**: Fixed-length sequences for deep learning models

## 🔄 Complete Processing Pipeline

### Step 1: Data Loading and Validation

**Input**: Raw CSV files in VitalDB format
**Output**: Loaded DataFrames with validation

```python
# Load all CSV files
csv_data = processor.load_csv_files()
```

**What happens:**
1. **File Discovery**: Scans directory for `.csv` files
2. **Data Loading**: Loads each CSV into pandas DataFrame
3. **Validation**: Checks file structure and basic integrity
4. **Logging**: Records loading statistics and any errors

**Expected Files:**
- `clinical_data.csv` - Patient demographics and outcomes
- `lab_data.csv` - Laboratory measurements over time
- `lab_parameters.csv` - Parameter definitions and metadata
- `clinical_parameters.csv` - Clinical parameter definitions (optional)
- `track_names.csv` - Vital sign track identifiers (optional)

### Step 2: Comprehensive Data Analysis

**Input**: Loaded DataFrames
**Output**: Analysis results and data quality metrics

```python
# Analyze each dataset
for file_name, df in csv_data.items():
    analysis = processor.analyze_csv_file(df, file_name)
```

**What happens:**
1. **Shape Analysis**: Records dimensions (rows × columns)
2. **Memory Usage**: Calculates memory footprint
3. **Data Types**: Identifies numeric, categorical, and datetime columns
4. **Missing Values**: Counts and percentages of missing data
5. **Duplicates**: Identifies duplicate records
6. **Completeness**: Calculates overall data completeness percentage
7. **Quality Issues**: Flags potential data quality problems

**Key Metrics Calculated:**
- **Completeness**: `(total_cells - missing_cells) / total_cells * 100`
- **Missing Rate**: Percentage of missing values per column
- **Data Type Distribution**: Count of each data type
- **Memory Usage**: Memory consumption in MB

### Step 3: Dataset Relationship Exploration

**Input**: Multiple DataFrames
**Output**: Relationship analysis and coverage statistics

```python
# Explore relationships between datasets
relationships = processor.explore_dataset_relationships()
```

**What happens:**
1. **Patient ID Mapping**: Identifies common patients across datasets
2. **Parameter Mapping**: Maps lab parameters to their definitions
3. **Coverage Calculation**: Calculates data coverage percentages
4. **Consistency Validation**: Ensures data consistency across files

**Key Relationships Analyzed:**
- **Case ID Overlap**: Common patients between clinical and lab data
- **Parameter Coverage**: Lab parameters with definitions available
- **Data Completeness**: Percentage of patients with complete data

**Example Output:**
```
Case ID overlap: 6,388 common patients (100.0% coverage)
Lab parameter coverage: 33 matched parameters (100.0% coverage)
```

### Step 4: Parameter Mapping Creation

**Input**: `lab_parameters.csv`
**Output**: Parameter interpretation dictionary

```python
# Create lab parameter mapping
param_mapping = processor.create_lab_parameter_mapping()
```

**What happens:**
1. **Parameter Indexing**: Creates index from parameter names
2. **Metadata Extraction**: Extracts descriptions, units, and categories
3. **Dictionary Creation**: Builds mapping for easy lookup
4. **Validation**: Ensures all required fields are present

**Mapping Structure:**
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

### Step 5: Integrated Clinical Dataset Creation

**Input**: Clinical data + Lab data + Parameter mappings
**Output**: Unified dataset with summary statistics

```python
# Create integrated clinical dataset
integrated_data = processor.create_integrated_clinical_dataset()
```

**Detailed Process:**

#### 5.1 Lab Data Summary Statistics
```python
lab_summary_stats = lab_df.groupby('caseid').agg({
    'dt': ['count', 'min', 'max'],      # Time range statistics
    'result': ['mean', 'std', 'min', 'max', 'count']  # Value statistics
})
```

**Creates per-patient summaries:**
- `dt_count`: Number of lab measurements
- `dt_min/max`: Time range of measurements
- `result_mean/std/min/max`: Statistical summaries of all lab values
- `result_count`: Total number of lab measurements

#### 5.2 Parameter-Specific Summaries
```python
key_parameters = ['wbc', 'hb', 'hct', 'plt', 'na', 'k', 'gluc', 'alb', 'cr', 'bun']

for param in key_parameters:
    param_data = lab_df[lab_df['name'] == param].groupby('caseid')['result'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ])
```

**Creates parameter-specific features:**
- `wbc_mean/std/min/max`: White blood cell statistics
- `hb_mean/std/min/max`: Hemoglobin statistics
- `na_mean/std/min/max`: Sodium statistics
- And so on for each key parameter...

#### 5.3 Data Merging
```python
# Merge clinical data with lab summaries
integrated_clinical = clinical_df.merge(lab_summary_stats, on='caseid', how='left')
integrated_clinical = integrated_clinical.merge(param_summary_df, on='caseid', how='left')
```

**Final Dataset Structure:**
```
caseid | age | sex | death_inhosp | dt_count | result_mean | wbc_mean | hb_mean | ...
123    | 65  | M   | 0           | 45       | 12.5       | 8.2      | 14.3    | ...
456    | 72  | F   | 1           | 38       | 15.2       | 7.8      | 12.1    | ...
```

### Step 6: Time-Series Sequence Creation

**Input**: Lab data + Clinical outcomes
**Output**: Fixed-length sequences for deep learning

```python
# Create time-series sequences
sequences, targets, patient_info = processor.create_time_series_sequences(
    max_length=48,
    min_lab_records=10
)
```

**Detailed Process:**

#### 6.1 Patient Selection
```python
# Select patients with sufficient lab data
patient_lab_counts = lab_df.groupby('caseid').size()
eligible_patients = patient_lab_counts[patient_lab_counts >= min_lab_records].index
```

**Criteria:**
- Minimum 10 lab records per patient
- Ensures sufficient temporal information
- Filters out patients with sparse data

#### 6.2 Sequence Creation
```python
for caseid in eligible_patients:
    patient_lab = lab_df[lab_df['caseid'] == caseid].sort_values('dt')
    
    # Create overlapping sequences
    for i in range(0, len(patient_lab) - max_length + 1, max_length // 2):
        sequence_lab = patient_lab.iloc[i:i + max_length]
```

**Sequence Generation:**
- **Fixed Length**: Each sequence has exactly 48 time points
- **Overlapping Windows**: Creates multiple sequences per patient
- **Temporal Ordering**: Maintains chronological order of measurements
- **Stride**: Half sequence length (24 time points) for overlap

#### 6.3 Data Pivoting
```python
# Pivot to get parameter columns
seq_pivot = sequence_lab.pivot_table(
    index='dt', 
    columns='name', 
    values='result', 
    aggfunc='mean'
)
```

**Transformation:**
```
BEFORE (Long Format):
caseid | dt  | name | result
123    | 100 | wbc  | 8.5
123    | 100 | hb   | 12.3
123    | 150 | wbc  | 9.2

AFTER (Wide Format):
dt   | wbc | hb  | ...
100  | 8.5 | 12.3| ...
150  | 9.2 | NaN | ...
```

#### 6.4 Missing Value Handling
```python
# Fill missing values
seq_pivot = seq_pivot.fillna(method='ffill').fillna(method='bfill')
```

**Missing Value Strategy:**
1. **Forward Fill**: Use last known value (clinically reasonable)
2. **Backward Fill**: Use next known value (for initial missing values)
3. **Padding**: Repeat last values for sequences shorter than required length

#### 6.5 Sequence Padding
```python
# Ensure consistent column structure
if seq_pivot.shape[0] < max_length:
    padding_needed = max_length - seq_pivot.shape[0]
    last_row = seq_pivot.iloc[-1:].copy()
    for _ in range(padding_needed):
        seq_pivot = pd.concat([seq_pivot, last_row])
```

**Padding Strategy:**
- **Short Sequences**: Pad with last known values
- **Consistent Length**: All sequences have exactly 48 time points
- **Parameter Consistency**: All sequences have same parameter columns

#### 6.6 Target Association
```python
# Get clinical outcome
clinical_info = clinical_df[clinical_df['caseid'] == caseid]
target = clinical_info['death_inhosp'].iloc[0]

sequences.append(seq_pivot.values)
targets.append(target)
patient_info.append({
    'caseid': caseid,
    'sequence_length': len(sequence_lab),
    'time_range': (sequence_lab['dt'].min(), sequence_lab['dt'].max())
})
```

**Target Variables:**
- **Mortality**: `death_inhosp` (0 = survived, 1 = died)
- **Metadata**: Patient ID, sequence length, time range

### Step 7: Data Saving and Persistence

**Input**: Processed datasets and sequences
**Output**: Saved files for future use

```python
# Save processed data
processor.save_integrated_data(output_dir)
```

**Files Created:**

#### 7.1 Integrated Clinical Data
**File**: `integrated_clinical_data.csv`
**Format**: CSV with all merged features
**Content**: Patient demographics + lab summaries + parameter-specific statistics

#### 7.2 Time-Series Sequences
**File**: `time_series_sequences.npz`
**Format**: Compressed NumPy arrays
**Content**:
- `sequences`: 3D array (N sequences × 48 time points × 33 parameters)
- `targets`: 1D array (binary outcomes for each sequence)
- `patient_info`: List of metadata dictionaries

#### 7.3 Parameter Mapping
**File**: `lab_parameter_mapping.json`
**Format**: JSON dictionary
**Content**: Parameter names mapped to descriptions, units, and categories

## 📊 Data Flow Summary

```
Raw CSV Files
    ↓ (Step 1: Load & Validate)
Loaded DataFrames
    ↓ (Step 2: Analyze Quality)
Data Quality Metrics
    ↓ (Step 3: Explore Relationships)
Relationship Analysis
    ↓ (Step 4: Create Mappings)
Parameter Mappings
    ↓ (Step 5: Integrate Data)
Integrated Clinical Dataset
    ↓ (Step 6: Create Sequences)
Time-Series Sequences
    ↓ (Step 7: Save Data)
Persistent Files
```

## 🎯 Output Data Characteristics

### Integrated Clinical Dataset
- **Shape**: (6,388 patients, 90+ features)
- **Features**: Demographics + lab summaries + parameter-specific statistics
- **Use Cases**: Traditional ML, feature analysis, patient stratification

### Time-Series Sequences
- **Shape**: (N sequences, 48 time points, 33 parameters)
- **Features**: Temporal lab measurements with missing value handling
- **Use Cases**: Deep learning, transformer models, sequence prediction

### Parameter Mapping
- **Content**: 33 lab parameters with clinical interpretations
- **Use Cases**: Model interpretation, clinical validation, feature engineering

## 🔧 Configuration Options

### Sequence Creation Parameters
- **max_length**: Sequence length (default: 48)
- **min_lab_records**: Minimum records per patient (default: 10)
- **max_patients**: Maximum patients to process (default: None)

### Missing Value Handling
- **Forward Fill**: Last known value
- **Backward Fill**: Next known value
- **Padding**: Last value repetition

### Data Quality Thresholds
- **Completeness**: Minimum data completeness percentage
- **Coverage**: Minimum parameter coverage percentage
- **Consistency**: Data consistency validation

## 🚀 Next Steps for Analysis

1. **Load Processed Data**: Use `load_processed_data()` for quick access
2. **Feature Engineering**: Extract temporal features and clinical ratios
3. **Model Training**: Train transformer models on sequences
4. **Evaluation**: Use clinical metrics for model assessment
5. **Interpretation**: Analyze attention weights and feature importance

This comprehensive processing pipeline transforms raw VitalDB data into analysis-ready formats suitable for both traditional machine learning and advanced deep learning approaches in clinical prediction tasks.
