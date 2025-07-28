# BostonML

A comprehensive machine learning project for Boston Housing Price Prediction using multiple algorithms and scaling techniques.

## 🚀 Features

- **Multiple Scaling**: MinMax, Standard, and Robust scaling comparison
- **15+ ML Algorithms**: Linear Regression, Random Forest, SVM, AdaBoost, etc.
- **Automated Tracking**: Performance registry across all model configurations
- **Hyperparameter Tuning**: GridSearch and RandomizedSearch optimization
- **Model Persistence**: Auto-save best performing models

## 📁 Structure

```
BostonML/
├── 01-03. BostonML_*.ipynb       # ML models with different scalers
├── 04. Analysis.ipynb            # Performance comparison
├── helperfunctions.py            # Model utilities
├── history_registry/             # Performance tracking
├── saved_models_*/               # Best models
└── Datasets/                     # Boston Housing data
```

## 🛠️ Quick Start

```bash
git clone https://github.com/yourusername/BostonML.git
cd BostonML
pip install -r requirements.txt
jupyter notebook
```

## 📊 Results

- **Best R²**: Tracked across 3 scalers × 15+ algorithms
- **Automated Registry**: CSV files track all model performance
- **Comprehensive Analysis**: Scaler impact and algorithm comparison

## 🎯 Key Notebooks

1. **01-03. BostonML_[Scaler].ipynb** - Run models with different scalers
2. **04. Analysis.ipynb** - Compare results and find best performers

Built with Python, Scikit-learn, Pandas, and Jupyter.