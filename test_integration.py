"""
Integration test – verifies environment and basic imports
"""
import torch
import sys
import os

print("="*60)
print("ATLAS AI - Project Integration Test")
print("="*60)

print(f"✓ PyTorch Version: {torch.__version__}")
print(f"✓ Python Path: {sys.executable}")
print(f"✓ Working Directory: {os.getcwd()}")

try:
    import numpy as np
    import pandas as pd
    print("✓ NumPy & Pandas installed")
except ImportError as e:
    print(f"✗ Missing: {e}")

# Check for model directory and base file
model_file = "src/models/traffic_base.py"
if os.path.exists(model_file):
    print(f"✓ Found {model_file}")
    try:
        from src.models.traffic_base import BaseTrafficAI
        print("✓ Successfully imported BaseTrafficAI")
    except Exception as e:
        print(f"✗ Import error: {e}")
else:
    print(f"✗ Missing {model_file}")
    print("  → Please create this file using the provided code.")

print("\n" + "="*60)
print("NEXT STEPS:")
print("="*60)
print("1. Run: python train_simple.py")
print("2. Verify training runs without errors")
print("3. Start extending with your team's features")
print("="*60)
