#!/usr/bin/env python3
"""
Test script to verify the EHR datasets environment is properly set up.
This script tests all major dependencies for time-series prediction, tabular data, and transformers.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing EHR Datasets Environment Setup")
    print("=" * 50)
    
    # Core data science
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✅ Core data science libraries (numpy, pandas, matplotlib, seaborn)")
    except ImportError as e:
        print(f"❌ Core data science libraries: {e}")
        return False
    
    # Machine learning
    try:
        import sklearn
        import scipy
        print("✅ Machine learning libraries (sklearn, scipy)")
    except ImportError as e:
        print(f"❌ Machine learning libraries: {e}")
        return False
    
    # Time series analysis
    try:
        import statsmodels
        import tslearn
        import darts
        import gluonts
        print("✅ Time series libraries (statsmodels, tslearn, darts, gluonts)")
    except ImportError as e:
        print(f"❌ Time series libraries: {e}")
        return False
    
    # Deep learning and transformers
    try:
        import torch
        import torchvision
        import transformers
        import pytorch_lightning as pl
        print("✅ Deep learning libraries (torch, transformers, pytorch-lightning)")
    except ImportError as e:
        print(f"❌ Deep learning libraries: {e}")
        return False
    
    # Tabular ML
    try:
        import catboost
        import xgboost as xgb
        import lightgbm as lgb
        print("✅ Tabular ML libraries (catboost, xgboost, lightgbm)")
    except ImportError as e:
        print(f"❌ Tabular ML libraries: {e}")
        return False
    
    # Healthcare-specific
    try:
        import lifelines
        import imblearn  # Note: actual import name is imblearn, not imbalanced_learn
        print("✅ Healthcare-specific libraries (lifelines, imbalanced-learn)")
    except ImportError as e:
        print(f"❌ Healthcare-specific libraries: {e}")
        return False
    
    # Jupyter and development
    try:
        import jupyter
        import jupyterlab
        print("✅ Jupyter development environment")
    except ImportError as e:
        print(f"❌ Jupyter development environment: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of key libraries"""
    print("\n🔧 Testing Basic Functionality")
    print("=" * 50)
    
    try:
        import numpy as np
        import pandas as pd
        import torch
        
        # Test NumPy
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.mean() == 3.0
        print("✅ NumPy basic operations")
        
        # Test Pandas
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert df.shape == (3, 2)
        print("✅ Pandas DataFrame operations")
        
        # Test PyTorch
        tensor = torch.tensor([1.0, 2.0, 3.0])
        assert tensor.sum().item() == 6.0
        print("✅ PyTorch tensor operations")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_project_structure():
    """Test that project structure is accessible"""
    print("\n📁 Testing Project Structure")
    print("=" * 50)
    
    try:
        import os
        from pathlib import Path
        
        # Check if we're in the right directory
        current_dir = Path.cwd()
        if current_dir.name == "ehr-datasets":
            print("✅ Correct project directory")
        else:
            print(f"⚠️  Current directory: {current_dir.name} (expected: ehr-datasets)")
        
        # Check key directories
        key_dirs = ['mimic-iii', 'mimic-iv', 'eICU', 'utils']
        for dir_name in key_dirs:
            if Path(dir_name).exists():
                print(f"✅ Found {dir_name}/ directory")
            else:
                print(f"⚠️  Missing {dir_name}/ directory")
        
        # Check key files
        key_files = ['requirements.txt', 'environment.yml', 'setup.py', 'README.md']
        for file_name in key_files:
            if Path(file_name).exists():
                print(f"✅ Found {file_name}")
            else:
                print(f"❌ Missing {file_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Project structure test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🏥 EHR Datasets Environment Test")
    print("=" * 50)
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print()
    
    # Run all tests
    tests = [
        test_imports,
        test_basic_functionality,
        test_project_structure
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 All {total} tests passed! Environment is ready for EHR datasets work.")
        print("\n🚀 You can now:")
        print("  • Run preprocessing notebooks in dataset directories")
        print("  • Use time-series analysis tools (Darts, GluonTS)")
        print("  • Apply transformer models for healthcare data")
        print("  • Perform tabular ML with CatBoost, XGBoost, LightGBM")
        print("  • Conduct survival analysis with Lifelines")
        return 0
    else:
        print(f"⚠️  {passed}/{total} tests passed. Some issues detected.")
        return 1

if __name__ == "__main__":
    exit(main())
