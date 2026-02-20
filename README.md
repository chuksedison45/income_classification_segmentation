# Income Classification and Customer Segmentation Machine Learning Project

This repository contains a Jupyter Notebook that solves a two-part data science problem for a retail client:
1. **Binary Classification**: Predict whether an individual earns more or less than $50,000 based on demographic and employment features.
2. **Customer Segmentation**: Group individuals into clusters for targeted marketing.

The dataset is a weighted census bureau dataset from the 1994-1995 Current Population Surveys (CPS) provided by the U.S. Census Bureau.

## Project Structure
- `census-bureau.data`: The input dataset (40 features + weight + label).
- `census-bureau.columns`: The input dataset columns
- `income_classification_notebook.ipynb`: Main Jupyter Notebook with all code, analysis, and results for income classification.
- `customer_segmentation_notebook.ipynb`: Jupyter Notebook for customer segmentation.
- `income_segmentation_notebook.py`: Python script for income segmentation.
- `income_classification_notebook.py`: Python script for income classification.
- `ML_TakeHomeProject_Report.pptx`: Presentation slides.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/chuksedison45/income_classification_segmentation.git
cd income_classification_segmentation
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
Launch Jupyter:
```bash
jupyter notebook .
```
Open `income_prediction_segmentation.ipynb` and execute all cells.

## Dependencies
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn
- jupyter
- scikit-learn-extra
- kmodes

All versions are specified in `requirements.txt`.

## Results Summary

### Classification
- Best model: **XGBoost** with SMOTE oversampling.
- Test set performance:
  - Accuracy: ~0.95
  - Precision: ~0.62
  - Recall: ~0.56
  - F1-score: ~0.69
  - ROC-AUC: ~0.9482
- Top predictive features: capital gains, age, education level, weeks worked.

### Segmentation
- **2 clusters** identified via K-Means on PCA-reduced data.
- Each cluster has distinct demographic profiles and income distributions, enabling tailored marketing strategies.

## Usage Notes
- The notebook handles missing values (coded as `?` or `Not in universe`) by imputation.
- Class imbalance is addressed using SMOTE during training.
- The clustering uses all features except the target and weight.

## License
This project is for educational/demonstration purposes.
