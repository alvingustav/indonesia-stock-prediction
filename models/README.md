# Pre-trained Models

Folder ini berisi model LSTM-GRU Hybrid yang sudah ditraining untuk prediksi saham Indonesia.

## Required Files

1. **stock_prediction_model.h5** - Model TensorFlow/Keras yang sudah ditraining
2. **scalers.pkl** - Feature dan target scalers (joblib dump)
3. **feature_columns.json** - Konfigurasi feature columns (auto-generated)

## Model Details

- **Architecture**: LSTM-GRU Hybrid dengan 145,751 parameters
- **Training Data**: 39,865 sequences dari 7 saham Indonesia
- **Features**: 22 technical indicators
- **Sequence Length**: 60 hari
- **Training Stocks**: BBCA, ASII, INDF, TLKM, BMRI, BBNI, IHSG

## Upload Instructions

### Method 1: Direct Upload di Codespaces
1. Buka VS Code di Codespaces
2. Drag & drop file ke folder models/
3. Commit: `git add models/ && git commit -m "Add models"`

### Method 2: Git LFS (Recommended untuk file besar)
Setup Git LFS
git lfs install
git lfs track ".h5" ".pkl"

Add dan commit
git add .gitattributes models/
git commit -m "Add pre-trained models with LFS"
git push

### Method 3: Script Helper
Upload dari local directory
python scripts/upload-models.py /path/to/your/models/


## Validation
Setelah upload, jalankan:
bash scripts/setup-models.sh
Model siap digunakan jika semua ✅ checks passed!
