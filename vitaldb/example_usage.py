#!/usr/bin/env python3
"""
VitalDB Processing Example

This script demonstrates how to use the VitalDBProcessor to transform
raw VitalDB CSV files into integrated clinical datasets and time-series
sequences suitable for deep learning models.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from vitaldb.vitaldb_clinical_process import VitalDBProcessor
import pandas as pd
import numpy as np

def main():
    """
    Main example function demonstrating VitalDB processing pipeline
    """
    print("🏥 VitalDB Clinical Data Processing Example")
    print("=" * 60)
    
    # Configuration - Update these paths for your setup
    data_dir = "/Users/nguyennghia/EHR/DATA/vital_files_subsets"
    output_dir = "/Users/nguyennghia/EHR/DATA/vital_files_subsets/processed"
    
    print(f"📁 Data directory: {data_dir}")
    print(f"💾 Output directory: {output_dir}")
    
    try:
        # Initialize processor
        print("\n1️⃣ Initializing VitalDB Processor...")
        processor = VitalDBProcessor(data_dir)
        
        # Step 1: Load CSV files
        print("\n2️⃣ Loading CSV files...")
        csv_data = processor.load_csv_files()
        print(f"   ✅ Loaded {len(csv_data)} CSV files")
        
        # Step 2: Analyze datasets
        print("\n3️⃣ Analyzing datasets...")
        for file_name, df in csv_data.items():
            analysis = processor.analyze_csv_file(df, file_name)
            print(f"   📊 {file_name}: {analysis['shape']} shape, {analysis['completeness']:.1f}% complete")
        
        # Step 3: Explore relationships
        print("\n4️⃣ Exploring dataset relationships...")
        relationships = processor.explore_dataset_relationships()
        
        if 'caseid_overlap' in relationships:
            overlap = relationships['caseid_overlap']
            print(f"   🔗 Case ID overlap: {overlap['common_caseids']:,} common patients")
            print(f"   📈 Coverage: {overlap['coverage']:.1f}% of clinical cases have lab data")
        
        # Step 4: Create parameter mapping
        print("\n5️⃣ Creating parameter mapping...")
        param_mapping = processor.create_lab_parameter_mapping()
        print(f"   🗺️  Created mapping for {len(param_mapping)} parameters")
        
        # Show sample parameters
        print("   📋 Sample parameters:")
        for i, (param, info) in enumerate(list(param_mapping.items())[:5]):
            print(f"      • {param}: {info.get('Description', 'N/A')} ({info.get('Unit', 'N/A')})")
        
        # Step 5: Create integrated clinical dataset
        print("\n6️⃣ Creating integrated clinical dataset...")
        integrated_data = processor.create_integrated_clinical_dataset()
        print(f"   📊 Integrated dataset shape: {integrated_data.shape}")
        
        # Show sample of integrated data
        print("   📋 Sample integrated data:")
        sample_cols = ['caseid', 'age', 'sex', 'death_inhosp']
        if all(col in integrated_data.columns for col in sample_cols):
            print(integrated_data[sample_cols].head().to_string(index=False))
        
        # Step 6: Create time-series sequences
        print("\n7️⃣ Creating time-series sequences...")
        sequences, targets, patient_info = processor.create_time_series_sequences(
            max_length=48,
            min_lab_records=10,
            max_patients=50  # Limit for example
        )
        
        print(f"   ⏰ Created {len(sequences)} sequences")
        if sequences:
            print(f"   📏 Sequence shape: {sequences[0].shape}")
            print(f"   🎯 Target distribution: {pd.Series(targets).value_counts().to_dict()}")
        
        # Step 7: Save processed data
        print("\n8️⃣ Saving processed data...")
        processor.save_integrated_data(output_dir)
        print(f"   💾 Saved to: {output_dir}")
        
        # Step 8: Print comprehensive summary
        print("\n9️⃣ Processing Summary...")
        summary = processor.get_data_summary()
        
        print(f"   📊 CSV files processed: {len(summary['csv_files'])}")
        print(f"   📈 Integrated dataset: {summary['integrated_clinical_shape']}")
        print(f"   ⏰ Time-series sequences: {summary['sequences_count']}")
        print(f"   🗺️  Parameter mappings: {summary['parameter_mapping_count']}")
        
        if 'target_distribution' in summary:
            print(f"   🎯 Target distribution: {summary['target_distribution']}")
        
        print("\n✅ VitalDB processing completed successfully!")
        print("\n🚀 Next steps:")
        print("   1. Load processed data for machine learning")
        print("   2. Train transformer models on time-series sequences")
        print("   3. Evaluate model performance on clinical outcomes")
        print("   4. Interpret results using parameter mappings")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("   Please check that the data directory contains the required CSV files")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise

def demonstrate_loaded_data():
    """
    Demonstrate how to load and use previously processed data
    """
    print("\n" + "="*60)
    print("🔄 Demonstrating Loaded Data Usage")
    print("="*60)
    
    output_dir = "/Users/nguyennghia/EHR/DATA/vital_files_subsets/processed"
    
    try:
        # Load previously processed data
        processor = VitalDBProcessor("dummy")  # Path not used for loading
        processor.load_processed_data(output_dir)
        
        # Access loaded data
        integrated_data = processor.integrated_clinical
        sequences = processor.sequences
        targets = processor.targets
        param_mapping = processor.lab_param_mapping
        
        if integrated_data is not None:
            print(f"📊 Loaded integrated data: {integrated_data.shape}")
            
            # Example analysis
            if 'death_inhosp' in integrated_data.columns:
                mortality_rate = integrated_data['death_inhosp'].mean()
                print(f"📈 Mortality rate: {mortality_rate:.2%}")
                
                # Analyze by age groups
                integrated_data['age_group'] = pd.cut(
                    integrated_data['age'], 
                    bins=[0, 50, 65, 80, 100], 
                    labels=['<50', '50-65', '65-80', '>80']
                )
                
                age_mortality = integrated_data.groupby('age_group')['death_inhosp'].agg(['count', 'mean'])
                print("\n📊 Mortality by age group:")
                print(age_mortality)
        
        if sequences is not None:
            print(f"\n⏰ Loaded sequences: {len(sequences)} sequences")
            print(f"📏 Sequence shape: {sequences[0].shape if sequences else 'No sequences'}")
            
            # Example sequence analysis
            if sequences:
                seq_lengths = [len(seq) for seq in sequences]
                print(f"📊 Sequence length stats:")
                print(f"   Mean: {np.mean(seq_lengths):.1f}")
                print(f"   Min: {min(seq_lengths)}")
                print(f"   Max: {max(seq_lengths)}")
        
        print("\n✅ Data loading demonstration complete!")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")

if __name__ == "__main__":
    # Run main processing example
    main()
    
    # Demonstrate loading previously processed data
    demonstrate_loaded_data()
