#!/usr/bin/env python3
"""
VitalDB Clinical Data Processing Module

This module provides comprehensive functions to process VitalDB clinical data
from raw CSV files to integrated clinical datasets and time-series sequences
suitable for deep learning models.

Author: EHR Datasets Project
Date: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class VitalDBProcessor:
    """
    VitalDB Clinical Data Processor
    
    This class handles the complete pipeline from raw VitalDB CSV files
    to integrated clinical datasets and time-series sequences.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the VitalDB processor
        
        Args:
            data_dir (str): Path to directory containing VitalDB CSV files
        """
        self.data_dir = Path(data_dir)
        self.csv_data = {}
        self.lab_param_mapping = {}
        self.integrated_clinical = None
        self.sequences = None
        self.targets = None
        self.patient_info = None
        
        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        logger.info(f"VitalDB Processor initialized with data directory: {self.data_dir}")
    
    def load_csv_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the data directory
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping filename to DataFrame
        """
        logger.info("Loading CSV files...")
        
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        logger.info(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        for csv_file in csv_files:
            file_path = self.data_dir / csv_file
            try:
                df = pd.read_csv(file_path)
                self.csv_data[csv_file] = df
                logger.info(f"Loaded {csv_file}: {df.shape[0]:,} rows × {df.shape[1]} columns")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {str(e)}")
        
        return self.csv_data
    
    def analyze_csv_file(self, df: pd.DataFrame, file_name: str) -> Dict:
        """
        Comprehensive analysis of a CSV file
        
        Args:
            df (pd.DataFrame): DataFrame to analyze
            file_name (str): Name of the file
            
        Returns:
            Dict: Analysis results
        """
        logger.info(f"Analyzing {file_name}...")
        
        analysis = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_summary': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'numeric_cols': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_cols': list(df.select_dtypes(include=['object']).columns),
            'datetime_cols': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Calculate completeness
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = sum(analysis['missing_summary'].values())
        analysis['completeness'] = ((total_cells - missing_cells) / total_cells) * 100
        
        return analysis
    
    def create_lab_parameter_mapping(self) -> Dict:
        """
        Create lab parameter mapping for interpretation
        
        Returns:
            Dict: Parameter mapping dictionary
        """
        if 'lab_parameters.csv' not in self.csv_data:
            logger.warning("lab_parameters.csv not found, creating empty mapping")
            return {}
        
        lab_params_df = self.csv_data['lab_parameters.csv']
        self.lab_param_mapping = lab_params_df.set_index('Parameter').to_dict('index')
        
        logger.info(f"Created lab parameter mapping for {len(self.lab_param_mapping)} parameters")
        return self.lab_param_mapping
    
    def explore_dataset_relationships(self) -> Dict:
        """
        Explore relationships between datasets
        
        Returns:
            Dict: Relationship analysis results
        """
        logger.info("Exploring dataset relationships...")
        
        relationships = {}
        
        # Check caseid overlap between clinical and lab data
        if 'clinical_data.csv' in self.csv_data and 'lab_data.csv' in self.csv_data:
            clinical_df = self.csv_data['clinical_data.csv']
            lab_df = self.csv_data['lab_data.csv']
            
            clinical_caseids = set(clinical_df['caseid'].unique())
            lab_caseids = set(lab_df['caseid'].unique())
            common_caseids = clinical_caseids.intersection(lab_caseids)
            
            relationships['caseid_overlap'] = {
                'clinical_caseids': len(clinical_caseids),
                'lab_caseids': len(lab_caseids),
                'common_caseids': len(common_caseids),
                'coverage': len(common_caseids) / len(clinical_caseids) * 100
            }
            
            logger.info(f"Case ID overlap: {len(common_caseids):,} common patients "
                       f"({len(common_caseids)/len(clinical_caseids)*100:.1f}% coverage)")
        
        # Check lab parameter names vs lab data names
        if 'lab_parameters.csv' in self.csv_data and 'lab_data.csv' in self.csv_data:
            lab_params_df = self.csv_data['lab_parameters.csv']
            lab_df = self.csv_data['lab_data.csv']
            
            lab_param_names = set(lab_params_df['Parameter'].unique())
            lab_data_names = set(lab_df['name'].unique())
            common_lab_names = lab_param_names.intersection(lab_data_names)
            
            relationships['lab_parameter_overlap'] = {
                'defined_parameters': len(lab_param_names),
                'data_parameters': len(lab_data_names),
                'matched_parameters': len(common_lab_names),
                'coverage': len(common_lab_names) / len(lab_data_names) * 100
            }
            
            logger.info(f"Lab parameter coverage: {len(common_lab_names)} matched parameters "
                       f"({len(common_lab_names)/len(lab_data_names)*100:.1f}% coverage)")
        
        return relationships
    
    def create_integrated_clinical_dataset(self) -> pd.DataFrame:
        """
        Create integrated clinical dataset by merging clinical and lab data
        
        Returns:
            pd.DataFrame: Integrated clinical dataset
        """
        logger.info("Creating integrated clinical dataset...")
        
        if 'clinical_data.csv' not in self.csv_data or 'lab_data.csv' not in self.csv_data:
            raise ValueError("Required CSV files (clinical_data.csv, lab_data.csv) not found")
        
        clinical_df = self.csv_data['clinical_data.csv']
        lab_df = self.csv_data['lab_data.csv']
        
        # Create lab data summary for each patient
        logger.info("Creating lab data summary statistics per patient...")
        
        lab_summary_stats = lab_df.groupby('caseid').agg({
            'dt': ['count', 'min', 'max'],  # Number of measurements, time range
            'result': ['mean', 'std', 'min', 'max', 'count']  # Statistical summary
        }).round(3)
        
        # Flatten column names
        lab_summary_stats.columns = ['_'.join(col).strip() for col in lab_summary_stats.columns]
        lab_summary_stats = lab_summary_stats.reset_index()
        
        logger.info(f"Lab summary shape: {lab_summary_stats.shape}")
        
        # Create parameter-specific summaries for key lab values
        logger.info("Creating parameter-specific summaries...")
        
        key_parameters = ['wbc', 'hb', 'hct', 'plt', 'na', 'k', 'gluc', 'alb', 'cr', 'bun']
        param_summaries = []
        
        for param in key_parameters:
            if param in lab_df['name'].values:
                param_data = lab_df[lab_df['name'] == param].groupby('caseid')['result'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(3)
                
                param_data.columns = [f'{param}_{col}' for col in param_data.columns]
                param_data = param_data.reset_index()
                param_summaries.append(param_data)
                
                logger.info(f"  {param}: {len(param_data)} patients with data")
        
        # Merge parameter summaries
        if param_summaries:
            param_summary_df = param_summaries[0]
            for df in param_summaries[1:]:
                param_summary_df = param_summary_df.merge(df, on='caseid', how='outer')
            
            logger.info(f"Parameter summary shape: {param_summary_df.shape}")
        else:
            param_summary_df = pd.DataFrame({'caseid': clinical_df['caseid']})
        
        # Merge clinical data with lab summaries
        logger.info("Merging clinical data with lab summaries...")
        
        self.integrated_clinical = clinical_df.merge(lab_summary_stats, on='caseid', how='left')
        self.integrated_clinical = self.integrated_clinical.merge(param_summary_df, on='caseid', how='left')
        
        logger.info(f"Integrated clinical dataset shape: {self.integrated_clinical.shape}")
        logger.info(f"Added lab summary features: {self.integrated_clinical.shape[1] - clinical_df.shape[1]}")
        
        return self.integrated_clinical
    
    def create_time_series_sequences(self, 
                                   max_length: int = 48, 
                                   min_lab_records: int = 10,
                                   max_patients: Optional[int] = None) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """
        Create time-series sequences for deep learning
        
        Args:
            max_length (int): Maximum sequence length
            min_lab_records (int): Minimum lab records per patient
            max_patients (Optional[int]): Maximum number of patients to process
            
        Returns:
            Tuple[List[np.ndarray], List[int], List[Dict]]: Sequences, targets, patient info
        """
        logger.info(f"Creating time-series sequences (max_length={max_length}, min_records={min_lab_records})...")
        
        if 'lab_data.csv' not in self.csv_data or 'clinical_data.csv' not in self.csv_data:
            raise ValueError("Required CSV files (lab_data.csv, clinical_data.csv) not found")
        
        lab_df = self.csv_data['lab_data.csv']
        clinical_df = self.csv_data['clinical_data.csv']
        
        # Select patients with sufficient lab data
        patient_lab_counts = lab_df.groupby('caseid').size()
        eligible_patients = patient_lab_counts[patient_lab_counts >= min_lab_records].index
        
        if max_patients:
            eligible_patients = eligible_patients[:max_patients]
        
        logger.info(f"Eligible patients: {len(eligible_patients):,}")
        
        # Filter lab data for eligible patients
        eligible_lab_data = lab_df[lab_df['caseid'].isin(eligible_patients)].copy()
        
        # Create sequences
        sequences = []
        targets = []
        patient_info = []
        
        for caseid in eligible_patients:
            # Get lab data for this patient
            patient_lab = eligible_lab_data[eligible_lab_data['caseid'] == caseid].copy()
            
            if len(patient_lab) < 5:  # Skip patients with too few records
                continue
                
            # Sort by time
            patient_lab = patient_lab.sort_values('dt')
            
            # Get clinical outcome
            clinical_info = clinical_df[clinical_df['caseid'] == caseid]
            if len(clinical_info) == 0:
                continue
                
            target = clinical_info['death_inhosp'].iloc[0]
            
            # Create sequences of fixed length
            for i in range(0, len(patient_lab) - max_length + 1, max_length // 2):
                sequence_lab = patient_lab.iloc[i:i + max_length]
                
                # Pivot to get parameter columns
                seq_pivot = sequence_lab.pivot_table(
                    index='dt', 
                    columns='name', 
                    values='result', 
                    aggfunc='mean'
                )
                
                # Fill missing values with forward fill
                seq_pivot = seq_pivot.fillna(method='ffill').fillna(method='bfill')
                
                # Ensure consistent column structure
                if seq_pivot.shape[0] < max_length:
                    # Pad with last values if sequence is too short
                    padding_needed = max_length - seq_pivot.shape[0]
                    last_row = seq_pivot.iloc[-1:].copy()
                    for _ in range(padding_needed):
                        seq_pivot = pd.concat([seq_pivot, last_row])
                
                # Take only the required length
                seq_pivot = seq_pivot.iloc[:max_length]
                
                sequences.append(seq_pivot.values)
                targets.append(target)
                patient_info.append({
                    'caseid': caseid,
                    'sequence_length': len(sequence_lab),
                    'time_range': (sequence_lab['dt'].min(), sequence_lab['dt'].max())
                })
        
        self.sequences = sequences
        self.targets = targets
        self.patient_info = patient_info
        
        logger.info(f"Created {len(sequences)} sequences")
        if sequences:
            logger.info(f"Sequence shape: {sequences[0].shape}")
        
        target_dist = pd.Series(targets).value_counts()
        logger.info(f"Target distribution: {target_dist.to_dict()}")
        
        return sequences, targets, patient_info
    
    def save_integrated_data(self, output_dir: str):
        """
        Save integrated clinical data and sequences
        
        Args:
            output_dir (str): Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving integrated data to {output_path}")
        
        # Save integrated clinical dataset
        if self.integrated_clinical is not None:
            clinical_path = output_path / 'integrated_clinical_data.csv'
            self.integrated_clinical.to_csv(clinical_path, index=False)
            logger.info(f"Saved integrated clinical data: {clinical_path}")
        
        # Save sequences as numpy arrays
        if self.sequences is not None:
            sequences_path = output_path / 'time_series_sequences.npz'
            np.savez_compressed(
                sequences_path,
                sequences=np.array(self.sequences),
                targets=np.array(self.targets),
                patient_info=self.patient_info
            )
            logger.info(f"Saved time-series sequences: {sequences_path}")
        
        # Save parameter mapping
        if self.lab_param_mapping:
            import json
            mapping_path = output_path / 'lab_parameter_mapping.json'
            with open(mapping_path, 'w') as f:
                json.dump(self.lab_param_mapping, f, indent=2)
            logger.info(f"Saved parameter mapping: {mapping_path}")
    
    def load_processed_data(self, data_dir: str):
        """
        Load previously processed data
        
        Args:
            data_dir (str): Directory containing processed data files
        """
        data_path = Path(data_dir)
        
        logger.info(f"Loading processed data from {data_path}")
        
        # Load integrated clinical data
        clinical_path = data_path / 'integrated_clinical_data.csv'
        if clinical_path.exists():
            self.integrated_clinical = pd.read_csv(clinical_path)
            logger.info(f"Loaded integrated clinical data: {self.integrated_clinical.shape}")
        
        # Load sequences
        sequences_path = data_path / 'time_series_sequences.npz'
        if sequences_path.exists():
            data = np.load(sequences_path, allow_pickle=True)
            self.sequences = data['sequences']
            self.targets = data['targets']
            self.patient_info = data['patient_info'].tolist()
            logger.info(f"Loaded sequences: {len(self.sequences)} sequences")
        
        # Load parameter mapping
        mapping_path = data_path / 'lab_parameter_mapping.json'
        if mapping_path.exists():
            import json
            with open(mapping_path, 'r') as f:
                self.lab_param_mapping = json.load(f)
            logger.info(f"Loaded parameter mapping: {len(self.lab_param_mapping)} parameters")
    
    def get_data_summary(self) -> Dict:
        """
        Get comprehensive data summary
        
        Returns:
            Dict: Data summary information
        """
        summary = {
            'csv_files': list(self.csv_data.keys()),
            'integrated_clinical_shape': self.integrated_clinical.shape if self.integrated_clinical is not None else None,
            'sequences_count': len(self.sequences) if self.sequences is not None else 0,
            'parameter_mapping_count': len(self.lab_param_mapping),
        }
        
        if self.sequences:
            summary['sequence_shape'] = self.sequences[0].shape
            summary['target_distribution'] = pd.Series(self.targets).value_counts().to_dict()
        
        return summary


def main():
    """
    Main function demonstrating the complete VitalDB processing pipeline
    """
    # Configuration
    data_dir = "/Users/nguyennghia/EHR/DATA/vital_files_subsets"
    output_dir = "/Users/nguyennghia/EHR/DATA/vital_files_subsets/processed"
    
    # Initialize processor
    processor = VitalDBProcessor(data_dir)
    
    try:
        # Step 1: Load CSV files
        print("Step 1: Loading CSV files...")
        csv_data = processor.load_csv_files()
        
        # Step 2: Analyze datasets
        print("\nStep 2: Analyzing datasets...")
        for file_name, df in csv_data.items():
            analysis = processor.analyze_csv_file(df, file_name)
            print(f"  {file_name}: {analysis['shape']} shape, {analysis['completeness']:.1f}% complete")
        
        # Step 3: Explore relationships
        print("\nStep 3: Exploring dataset relationships...")
        relationships = processor.explore_dataset_relationships()
        
        # Step 4: Create parameter mapping
        print("\nStep 4: Creating parameter mapping...")
        param_mapping = processor.create_lab_parameter_mapping()
        
        # Step 5: Create integrated clinical dataset
        print("\nStep 5: Creating integrated clinical dataset...")
        integrated_data = processor.create_integrated_clinical_dataset()
        
        # Step 6: Create time-series sequences
        print("\nStep 6: Creating time-series sequences...")
        sequences, targets, patient_info = processor.create_time_series_sequences(
            max_length=48, 
            min_lab_records=10,
            max_patients=100
        )
        
        # Step 7: Save processed data
        print("\nStep 7: Saving processed data...")
        processor.save_integrated_data(output_dir)
        
        # Step 8: Print summary
        print("\nStep 8: Processing Summary...")
        summary = processor.get_data_summary()
        print(f"  CSV files processed: {len(summary['csv_files'])}")
        print(f"  Integrated dataset: {summary['integrated_clinical_shape']}")
        print(f"  Time-series sequences: {summary['sequences_count']}")
        print(f"  Parameter mappings: {summary['parameter_mapping_count']}")
        
        print("\n✅ VitalDB processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
