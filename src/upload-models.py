#!/usr/bin/env python3
"""
Helper script untuk upload model files ke Codespaces
"""

import os
import shutil
from pathlib import Path
import argparse

def upload_models(source_dir):
    """Upload model files dari local ke models directory"""
    
    models_dir = Path("models")
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"❌ Source directory tidak ditemukan: {source_path}")
        return False
    
    # Ensure models directory exists
    models_dir.mkdir(exist_ok=True)
    
    # Expected model files
    expected_files = [
        "stock_prediction_model.h5",
        "scalers.pkl"
    ]
    
    copied_files = []
    
    for file_name in expected_files:
        source_file = source_path / file_name
        target_file = models_dir / file_name
        
        if source_file.exists():
            try:
                shutil.copy2(source_file, target_file)
                file_size = target_file.stat().st_size
                size_mb = file_size / (1024 * 1024)
                print(f"✅ Copied {file_name} ({size_mb:.1f} MB)")
                copied_files.append(file_name)
            except Exception as e:
                print(f"❌ Error copying {file_name}: {e}")
        else:
            print(f"⚠️  File tidak ditemukan: {source_file}")
    
    if copied_files:
        print(f"\n🎉 Successfully uploaded {len(copied_files)} files:")
        for file_name in copied_files:
            print(f"   - {file_name}")
        
        print("\n📋 Next steps:")
        print("1. git add models/")
        print("2. git commit -m 'Add pre-trained models'")
        print("3. git push")
        
        return True
    else:
        print("\n❌ No files were uploaded")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model files to models directory")
    parser.add_argument("source_dir", help="Source directory containing model files")
    
    args = parser.parse_args()
    
    print("📦 Indonesia Stock Prediction - Model Upload")
    print("=" * 50)
    
    success = upload_models(args.source_dir)
    
    if success:
        print("\n✅ Upload completed successfully!")
    else:
        print("\n❌ Upload failed!")
