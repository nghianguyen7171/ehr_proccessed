"""
VitalDB Clinical Data Processing Package

This package provides comprehensive functions to process VitalDB clinical data
from raw CSV files to integrated clinical datasets and time-series sequences
suitable for deep learning models.

Main Components:
- VitalDBProcessor: Main processing class
- Data loading and analysis functions
- Time-series sequence creation
- Integration utilities

Example Usage:
    from vitaldb import VitalDBProcessor
    
    processor = VitalDBProcessor("/path/to/vitaldb/csv/files")
    integrated_data = processor.create_integrated_clinical_dataset()
    sequences, targets, patient_info = processor.create_time_series_sequences()
"""

from .vitaldb_clinical_process import VitalDBProcessor

__version__ = "1.0.0"
__author__ = "EHR Datasets Project"
__email__ = "contact@ehr-datasets.org"

__all__ = [
    "VitalDBProcessor",
]
