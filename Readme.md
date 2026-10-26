# Dry Bean Classification Using Machine Learning

## 1. Project Overview

This project compares five supervised machine learning classification algorithms to predict the class of a dry bean using its measured physical characteristics. The objective is to evaluate the models using multiple performance metrics and identify the best-performing model on the held-out test dataset.

The five algorithms evaluated are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

A Streamlit web application allows users to upload a labelled test dataset, select a trained model, and view its evaluation metrics and classification results.

## 2. Problem Statement

Dry bean varieties have different physical characteristics, including area, perimeter, shape, and dimensions. Classifying bean varieties based on these measurements can support automated and consistent classification.

The goal of this project is to build and compare multiple machine learning classification models using the Dry Bean dataset. The models are evaluated using Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).

## 3. Dataset Description

**Dataset name:** Dry Bean Dataset

**Source:** UCI Machine Learning Repository

**Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean

The dataset contains measurements of dry bean samples belonging to seven different classes. It includes 13,611 instances and 16 numerical input features.

The features describe the physical characteristics of the beans, such as area, perimeter, major axis length, minor axis length, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, and shape factors.

**Target variable:** `Class`

**Problem type:** Multiclass classification

The target variable represents the bean variety to be predicted from the input measurements.

## 4. Tools and Technologies

- Python
- Pandas and NumPy for data handling
- Scikit-learn for preprocessing, model training, and evaluation
- Matplotlib and Seaborn for visualisation
- Joblib for saving and loading trained models
- Jupyter Notebook for experimentation
- Streamlit for the interactive web application
- GitHub for source-code version control and project hosting
- Streamlit Community Cloud for application deployment

## 5. Methodology

The project follows these steps:

1. Load the Dry Bean dataset.
2. Inspect the dataset and identify the input features and target variable.
3. Perform a stratified train-test split using an 80:20 ratio and random state 42.
4. Apply the required preprocessing through Scikit-learn pipelines.
5. Train five machine learning classification models on the same training dataset.
6. Evaluate each model using the same held-out test dataset.
7. Compare the results using six performance metrics.
8. Visualise and save the model comparison results.
9. Save the trained models as Joblib files.
10. Develop a Streamlit application to evaluate the saved models using an uploaded labelled CSV file.

The saved models are loaded by the Streamlit application. Model training is performed in the notebook rather than each time a user evaluates a model in the application.

## 6. Machine Learning Models

### 6.1 Logistic Regression

Logistic Regression is used as a classification baseline. The implementation uses feature scaling and a higher iteration limit to support model convergence.

### 6.2 Decision Tree

Decision Tree learns classification rules from the input features. It can model nonlinear relationships and provides an interpretable tree-based approach.

### 6.3 K-Nearest Neighbors (KNN)

KNN predicts a class based on the nearest training samples. Feature scaling is applied because the algorithm relies on distances between observations.

### 6.4 Gaussian Naive Bayes

Gaussian Naive Bayes applies Bayes' theorem with a Gaussian likelihood assumption for the input features. It provides a probabilistic classification approach.

### 6.5 Support Vector Machine (SVM)

SVM learns decision boundaries to distinguish between classes. Feature scaling is applied to help ensure that differences in feature magnitudes do not dominate the model.

## 7. Evaluation Metrics

The following metrics are used to compare the models:

- **Accuracy:** Proportion of test instances classified correctly.
- **AUC:** Measures the model's ability to distinguish classes using ROC-based evaluation.
- **Precision:** Measures the correctness of positive class predictions, aggregated across classes.
- **Recall:** Measures how many actual instances are identified, aggregated across classes.
- **F1-Score:** Harmonic mean of Precision and Recall.
- **MCC:** Matthews Correlation Coefficient, a measure of prediction quality that accounts for correct and incorrect classifications.

Precision, Recall, F1-Score, and AUC are evaluated using the multiclass approach implemented in the experiment. The reported values should be interpreted according to the notebook's metric configuration.

## 8. Model Comparison Results

All models were evaluated using the same held-out test dataset. The following values are from the executed experiment and are rounded to four decimal places.

| Model | Accuracy | AUC | Precision | Recall | F1-Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9948 | 0.9349 | 0.9314 | 0.9329 | 0.9042 |
| Decision Tree | 0.8884 | 0.9531 | 0.9038 | 0.9041 | 0.9036 | 0.8652 |
| KNN | 0.9166 | 0.9833 | 0.9320 | 0.9271 | 0.9293 | 0.8992 |
| Gaussian Naive Bayes | 0.7639 | 0.9672 | 0.7744 | 0.7694 | 0.7677 | 0.7154 |
| Support Vector Machine (SVM) | 0.9221 | 0.9949 | 0.9359 | 0.9330 | 0.9343 | 0.9059 |

## 9. Model-wise Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 92.07% and an F1-Score of 0.9329. It performed strongly across the evaluation metrics and provided a competitive baseline for the classification task.

### Decision Tree

Decision Tree achieved an accuracy of 88.84% and an F1-Score of 0.9036. Its performance was lower than that of Logistic Regression, KNN, and SVM on this test dataset.

### K-Nearest Neighbors

KNN achieved an accuracy of 91.66% and an F1-Score of 0.9293. It performed competitively, although its overall results were slightly below those of SVM and Logistic Regression.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 76.39% and an F1-Score of 0.7677, the lowest values among the five models for these two metrics. This suggests that its assumptions may be less suitable for representing the patterns in this dataset than the other tested approaches.

### Support Vector Machine

SVM achieved the highest accuracy of 92.21%, AUC of 0.9949, Precision of 0.9359, Recall of 0.9330, F1-Score of 0.9343, and MCC of 0.9059. It performed best across all six reported metrics in this experiment.

## 10. Overall Best Model

Based on the evaluation results, **Support Vector Machine (SVM)** was the best-performing model among the five evaluated algorithms.

It achieved an accuracy of 92.21% and an F1-Score of 0.9343, along with the highest reported AUC, Precision, Recall, and MCC.

Logistic Regression was a close second, with an accuracy of 92.07% and an F1-Score of 0.9329. KNN also performed competitively. Decision Tree produced moderate results, while Gaussian Naive Bayes had the lowest accuracy and F1-Score.

These findings apply to the dataset, train-test split, preprocessing, and model configurations used in this experiment. Further validation would be required before generalising the conclusions to other datasets.

## 11. Streamlit Web Application

The Streamlit application provides an interface for evaluating the saved machine learning models.

### Application features

- Upload a labelled CSV test dataset.
- Select one of the five trained classification models.
- Calculate and display Accuracy, AUC, Precision, Recall, F1-Score, and MCC.
- Display a confusion matrix or classification report.
- Compare model performance by selecting different models.
- Download evaluation results when supported by the application.

The uploaded CSV must contain the same input features used during training and the target column `Class`. The application evaluates the selected saved model against the target labels in the uploaded file.

### Live application

**Streamlit App URL:** ADD_YOUR_STREAMLIT_APP_URL_HERE

## 12. GitHub Repository

The repository contains the notebook, application source code, dependencies, saved models, test dataset, and project documentation.

**GitHub Repository URL:** ADD_YOUR_GITHUB_REPOSITORY_URL_HERE

### Project structure

```text
dry-bean-classifier/
├── app.py
├── requirements.txt
├── README.md
├── Dry_Bean_Classification_Assignment.ipynb
├── test_data.csv
├── metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 13. Installation and Execution

### Prerequisites

- Python 3
- pip
- The project files and saved model files

### Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

### Run the Streamlit application

```bash
python3 -m streamlit run app.py
```

After the application starts, open the local URL displayed in the terminal, usually `http://localhost:8501`.

### Using the application

1. Open the Streamlit application.
2. Upload a labelled test CSV file containing the required input features and `Class` target column.
3. Select a trained model from the dropdown.
4. Review the evaluation metrics and confusion matrix or classification report.
5. Select another model to inspect its results.

## 14. Conclusion

This project implemented and evaluated five supervised machine learning algorithms for dry bean classification. The models were assessed using Accuracy, AUC, Precision, Recall, F1-Score, and MCC on the same held-out test dataset.

SVM achieved the strongest overall performance, followed closely by Logistic Regression and KNN. The experiment demonstrates the value of comparing multiple classification algorithms using consistent evaluation data and multiple metrics rather than relying on accuracy alone.

The Streamlit application makes the trained models accessible through a simple interface and enables users to evaluate them using labelled test data.

## 15. References

1. UCI Machine Learning Repository. *Dry Bean Dataset*. https://archive.ics.uci.edu/dataset/602/dry+bean
2. Scikit-learn documentation. https://scikit-learn.org/
3. Streamlit documentation. https://docs.streamlit.io/
4. Joblib documentation. https://joblib.readthedocs.io/

