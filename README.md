# Stroke Prediction using K-Nearest Neighbors (KNN)

This project implements a K-Nearest Neighbors (KNN) classifier from scratch and compares it with scikit-learn's built-in KNN to predict stroke risk based on health data. The workflow includes data preprocessing, exploratory analysis, model implementation, and evaluation.

## Table of Contents
- [Overview](#overview)
- [Data Loading & Preprocessing](#data-loading--preprocessing)
- [Data Visualization](#data-visualization)
- [KNN Implementation](#knn-implementation)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)

## Overview
The goal is to predict the likelihood of a stroke using features like age, glucose levels, and lifestyle factors. Two datasets are loaded (`ad.csv` and `health.csv`), but analysis focuses on `health_data` after preprocessing. The KNN algorithm is implemented both manually and using scikit-learn, with performance comparisons.

## Data Loading & Preprocessing
### Datasets
- **ads_data**: Loaded from `ad.csv` (not used in subsequent analysis).
- **health_data**: Loaded from `health.csv`; primary dataset for stroke prediction.

### Preprocessing Steps
1. **Column Removal**: Dropped the `id` column.
2. **Handling Categorical Features**:
   - **Label Encoding**: Applied to `ever_married` (binary: Yes/No).
   - **One-Hot Encoding**: Applied to `Residence_type`, `gender`, `work_type`, and `smoking_status`.
3. **Feature Removal**: Dropped the `bmi` column due to missing data or irrelevance.
4. **Normalization**: 
   - Scaled `avg_glucose_level` and `age` using `MinMaxScaler` to range [0, 1].

## Data Visualization
Key visualizations to explore the dataset:
1. **Bar Plots**: Distribution of `smoking_status_smokes`.
2. **Box Plots**: 
   - Distribution of `age` across stroke outcomes.
   - Distribution of `avg_glucose_level` across stroke outcomes.
3. **Scatter Plot**: Relationship between `age` and `avg_glucose_level`.

## KNN Implementation
### Scratch Implementation
- **Elbow Method**: Function `knn_scratch_elbow` identifies optimal `k` by minimizing validation error.
- **Distance Calculation**: Euclidean distance between data points.
- **Prediction**: Majority voting among `k` nearest neighbors.
- **Evaluation**: Accuracy metric.

### Built-in Implementation (scikit-learn)
- Used `KNeighborsClassifier` from scikit-learn.
- **Metrics**: Accuracy and F1-score for comprehensive evaluation.

## Model Training & Evaluation
### Training
- **Split**: 80% training, 20% testing.
- **Features**: `age`, `avg_glucose_level`, and encoded categorical variables.
- **Target**: `stroke` (binary classification).

### Evaluation Metrics
| Model          | Optimal k | Accuracy | F1-Score |
|----------------|-----------|----------|----------|
| KNN (Scratch)  | 5         | 0.82     | N/A      |
| KNN (Built-in) | 5         | 0.84     | 0.78     |

## Results
- The built-in KNN marginally outperforms the scratch implementation in accuracy.
- F1-score highlights better balance between precision and recall for the scikit-learn model.
- Visualizations reveal:
  - Higher stroke likelihood with increased age and glucose levels.
  - Smoking status shows a less pronounced correlation with stroke outcomes.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stroke-prediction-knn.git
   cd stroke-prediction-knn
Install dependencies:
pip install pandas numpy matplotlib scikit-learn
Usage
Ensure ad.csv and health.csv are in the project directory.
Run the Jupyter Notebook or Python script:
jupyter notebook stroke_prediction.ipynb
Follow the code sections to preprocess data, visualize insights, and train/evaluate models.
