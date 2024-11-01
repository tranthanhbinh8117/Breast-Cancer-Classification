# Breast Cancer Classification Using Machine Learning

This repository contains a machine learning project focused on classifying breast cancer as either malignant or benign using various machine learning algorithms. The goal is to leverage data-driven models to assist in the early detection and diagnosis of breast cancer, potentially aiding healthcare professionals in making informed decisions.

## Overview

Breast cancer is one of the most common cancers among women worldwide. Early detection and accurate diagnosis are crucial for effective treatment. In this project, we use machine learning techniques to classify breast cancer tumors based on features extracted from breast tissue samples.

## Project Workflow

1. **Data Collection**:
   - The dataset used is the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)), which contains features computed from digitized images of fine needle aspirate (FNA) of breast masses.
   - The dataset includes features such as radius, texture, perimeter, area, smoothness, compactness, and more.

2. **Data Preprocessing**:
   - Handling missing data, if any.
   - Feature scaling to ensure that all features contribute equally to the model.
   - Splitting the data into training and testing sets.

3. **Exploratory Data Analysis (EDA)**:
   - Visualizing the distribution of features.
   - Understanding the correlation between different features.
   - Identifying any patterns or insights that could help in the classification task.

4. **Model Selection**:
   - Implementing and comparing multiple machine learning models, including:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest
     - k-Nearest Neighbors (k-NN)
     - Gradient Boosting
   - Using cross-validation to select the best-performing model.

5. **Model Evaluation**:
   - Evaluating the models using metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
   - Analyzing confusion matrices to understand the performance of the models.
   - Tuning hyperparameters to optimize model performance.

6. **Deployment (Optional)**:
   - Deploying the best model using a web interface or API, allowing users to input data and receive predictions.

## Getting Started

### Prerequisites

Ensure you have Python installed, along with the following libraries:

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required libraries using:
```bash
pip install -r requirements.txt
```

### Running the Project

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/kientech/Breast-Cancer-Classification-Using-Machine-Learning
   cd Breast-Cancer-Classification-Using-Machine-Learning
   ```

2. **Explore the Data**:
   - The data is stored in the `data/` directory.
   - Use the Jupyter notebooks or scripts in the `notebooks/` or `scripts/` directory to explore and preprocess the data.

3. **Train the Models**:
   - Run the `model_training.ipynb` notebook to train and evaluate the models.
   - The best model will be saved in the `models/` directory.

4. **Make Predictions**:
   - Use the trained model to make predictions on new data.

## Results

- The best-performing model achieved an accuracy of **XX%**, with a precision of **XX%**, recall of **XX%**, and an AUC-ROC score of **XX**.
- The model is effective at distinguishing between malignant and benign tumors, providing a valuable tool for medical diagnosis.

## Contributions

Contributions are welcome! If you have any suggestions, improvements, or new features to add, feel free to open a pull request or an issue.
