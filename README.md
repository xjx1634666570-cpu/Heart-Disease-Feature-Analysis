# Heart Disease Risk Factor Analysis Dashboard

## 1. Project Overview

This project explores the relationships between demographic, clinical, and lifestyle variables and heart disease status.

An interactive web application is developed using **Streamlit**, **Pandas**, **NumPy**, **Plotly**, and **Scikit-Learn**. The dashboard allows users to explore the dataset, visualize linear correlations, examine nonlinear feature importance using machine learning, and dynamically investigate how data preprocessing and feature engineering affect analytical results.

The project addresses the following research questions:

1. Which selected variables have the strongest linear correlations with heart disease status?
2. Why might Pearson correlation fail to capture complex medical relationships, and how can nonlinear machine learning models, such as Random Forest, reveal additional predictive patterns?
3. How do medical binning and missing-data handling affect the analytical results?
4. How does class imbalance influence the interpretation of model performance?

## 2. Dataset

The dataset, `heart_disease.csv`, contains approximately 10,000 patient records.

### 2.1 Features

The analysis focuses on the following variables:

- **Clinical and Demographic Variables:** BMI, Blood Pressure, Homocysteine Level, Triglyceride Level, Cholesterol Level, Fasting Blood Sugar, Age, and Gender.
- **Lifestyle and Health Indicators:** Smoking, Alcohol Consumption, and High Blood Pressure.

### 2.2 Target Variable

The target variable is `Heart Disease Status`, which contains two classes:

- `Yes`: The patient has heart disease.
- `No`: The patient does not have heart disease.

### 2.3 Data Quality Considerations

The dataset presents two important challenges:

- **Class Imbalance:** Approximately 80% of records belong to the `No` class, while 20% belong to the `Yes` class.
- **Missing Values:** The `Alcohol Consumption` variable has a missing rate of approximately 26%.

These characteristics are considered during data preprocessing, visualization, and model development.

## 3. Technologies and Requirements

The project uses Python and the following libraries:

| Technology | Purpose |
|---|---|
| Streamlit | Interactive web application framework |
| Pandas | Data loading, cleaning, and manipulation |
| NumPy | Numerical computation |
| Plotly | Interactive charts and visualizations |
| Scikit-Learn | Random Forest modeling and feature importance analysis |

### 3.1 Installation

Install the required dependencies using pip:

```bash
pip install streamlit pandas numpy plotly scikit-learn
```

Alternatively, create a `requirements.txt` file with the following contents:

```text
streamlit
pandas
numpy
plotly
scikit-learn
```

Then install all dependencies:

```bash
pip install -r requirements.txt
```

## 4. How to Run the Application

Follow these steps to launch the dashboard.

1. Download or clone the project files.
2. Ensure that `app.py` and `heart_disease.csv` are located in the same directory.
3. Open a terminal in the project directory.
4. Run the following command:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL displayed in the terminal, typically:

   `http://localhost:8501`

The application also supports uploading a CSV file through the sidebar.

## 5. Project Structure

The project directory is organized as follows:

```text
HeartDiseaseProject/
├── app.py
├── heart_disease.csv
├── README.md
└── requirements.txt
```

## 6. Dashboard Features

### 6.1 Dynamic Data Handling

The sidebar provides options for adjusting data preprocessing and analysis.

**Alcohol Consumption Toggle**

Because the `Alcohol Consumption` variable has approximately 26% missing values, users can choose whether to include it in the analysis.

- When included, complete-case analysis may reduce the number of valid records to approximately 7,200.
- When excluded, the analysis may retain approximately 9,900 valid records.

These figures are approximate and may vary depending on the dataset and preprocessing settings.

**Medical Binning**

Continuous variables, such as Age, BMI, and Blood Pressure, can be converted into ordinal risk levels ranging from 1 to 5.

This transformation allows users to investigate whether categorizing continuous measurements helps reveal potential nonlinear or threshold-based relationships.

### 6.2 Dataset Overview

The dashboard displays summary metrics that update according to the selected variables and preprocessing options:

- **Total Records:** Number of records in the dataset.
- **Valid Records:** Number of records remaining after data cleaning.
- **Heart Disease Cases:** Number of records with a positive heart disease status.
- **Heart Disease Rate:** Proportion of valid records classified as having heart disease.

These metrics help users understand the effects of missing-data removal and changes in the analyzed sample.

### 6.3 Linear Analysis: Pearson Correlation

A horizontal bar chart displays the Pearson correlation coefficient between each selected feature and heart disease status.

- **Red bars:** Positive correlations.
- **Blue bars:** Negative correlations.

Pearson correlation measures the strength and direction of a linear relationship. It provides a useful starting point for identifying simple associations between individual variables and the target.

However, a low correlation coefficient does not necessarily indicate that a variable lacks predictive value.

### 6.4 Nonlinear Analysis: Random Forest Feature Importance

The dashboard trains a Random Forest classifier to examine the predictive importance of the selected features.

The model uses:

```python
class_weight="balanced"
```

This setting adjusts class weights to help account for the unequal representation of positive and negative cases.

A green bar chart displays the feature importance scores calculated by the model. Unlike Pearson correlation, Random Forest can capture certain nonlinear relationships and interactions between variables.

The analysis has identified Age as a dominant predictor, followed by BMI and Cholesterol in the reported results.

These findings should be interpreted as dataset-specific predictive patterns rather than proof of clinical importance or causation.

### 6.5 Heart Disease Status Distribution and Correlation Heatmap

**Heart Disease Status Distribution**

A donut chart visualizes the proportions of positive and negative heart disease cases in the cleaned dataset.

This visualization helps users assess class imbalance and understand the composition of the analyzed sample.

**Correlation Heatmap**

A heatmap displays pairwise correlations among the selected variables.

It helps identify potential relationships and overlap between features, which may provide context for interpreting the correlation chart and Random Forest feature importance scores.

### 6.6 Cleaned Dataset Preview

Users can enable the dataset preview option in the sidebar to inspect the processed records.

The preview may include:

- Selected analysis variables.
- Cleaned and encoded values.
- Records remaining after missing-value removal.
- Transformed variables when medical binning is enabled.

This feature makes the data preparation process more transparent and allows users to examine how preprocessing settings affect the final dataset.

## 7. Data Preparation and Feature Engineering

The application performs the following preprocessing steps:

1. **Feature Selection:** Selects the specified demographic, clinical, and lifestyle variables together with the target variable.
2. **Text Cleaning:** Removes leading and trailing whitespace from text values.
3. **Target Encoding:** Converts `Heart Disease Status` from `Yes` and `No` to binary values of 1 and 0.
4. **Categorical Encoding:** Encodes supported binary categorical variables into numerical representations.
5. **Numeric Conversion:** Converts continuous measurements to numeric types for statistical analysis and modeling.
6. **Missing-Value Handling:** Applies complete-case analysis by removing records with missing values in the selected analysis variables.
7. **Medical Binning:** Optionally transforms continuous variables into ordinal risk categories ranging from 1 to 5.
8. **Model Training:** Trains a Random Forest classifier and extracts feature importance scores.

The preprocessing options allow users to explore how variable selection, missing-data handling, and feature engineering influence the resulting analysis.

## 8. Key Analytical Considerations

### 8.1 Limitations of Linear Correlation

Pearson correlation measures linear associations between two variables. However, medical risk factors may have more complex relationships with disease outcomes.

For example:

- **Threshold Effects:** Risk may increase more rapidly after a particular measurement exceeds a threshold.
- **Nonlinear Relationships:** Risk may follow a U-shaped or other nonlinear pattern.
- **Feature Interactions:** The predictive value of one variable may depend on the values of other variables.

Consequently, low Pearson correlation coefficients, including values below 0.05, do not automatically mean that the corresponding variables are unimportant for prediction.

### 8.2 The Potential of Machine Learning

Random Forest can model nonlinear relationships and interactions that simple linear correlation does not capture directly.

In the reported analysis, Age emerged as the dominant predictor, followed by BMI and Cholesterol.

These results illustrate how machine learning can complement traditional correlation analysis. However, feature importance depends on the dataset, preprocessing choices, model configuration, and evaluation methodology. It does not independently establish causal relationships or clinical validity.

### 8.3 Class Imbalance

The dataset contains approximately 80% negative cases and 20% positive cases.

Under this distribution, a classifier that always predicts `No` could achieve approximately 80% accuracy while failing to identify any positive cases.

The project uses `class_weight="balanced"` in the Random Forest model to account for class imbalance during training.

For future predictive evaluation, the following metrics should be considered alongside accuracy:

- **Recall:** Measures the proportion of actual heart disease cases correctly identified.
- **Precision:** Measures the proportion of predicted positive cases that are actually positive.
- **F1-Score:** Combines precision and recall into a single metric.
- **Confusion Matrix:** Shows true positives, true negatives, false positives, and false negatives.

These metrics provide a more informative assessment of performance when the target classes are imbalanced.

## 9. Limitations

Several limitations should be considered when interpreting the results.

- **Exploratory Analysis:** The dashboard is designed for educational and exploratory purposes rather than clinical decision-making.
- **No Causal Inference:** Correlations and feature importance scores do not establish causation.
- **Dataset Quality:** Weak linear correlations and relatively low feature importance scores may reflect noise, limited signal, data-generation characteristics, or other dataset limitations.
- **Missing-Value Bias:** Complete-case analysis can reduce sample size and introduce bias if records with missing values differ systematically from complete records.
- **Model Evaluation:** Feature importance alone does not establish predictive performance on unseen data. A separate test set and appropriate evaluation metrics are needed.
- **Feature Importance Limitations:** Random Forest importance scores can be affected by correlated predictors and the selected importance calculation method.
- **Clinical Validity:** The model has not been established as a validated clinical prediction system.

## 10. Future Improvements

Potential extensions include:

1. **Comparative Predictive Modeling:** Compare Logistic Regression, Random Forest, and other classification algorithms using a consistent training and testing procedure.
2. **Model Evaluation:** Report Recall, Precision, F1-Score, ROC-AUC, and confusion matrices on held-out data.
3. **SHAP Interpretability:** Use SHAP values to examine how individual features contribute to model predictions.
4. **Missing-Value Imputation:** Investigate methods such as KNN imputation to retain more records when `Alcohol Consumption` is included.
5. **Threshold Optimization:** Evaluate classification thresholds to explore the trade-off between identifying positive cases and generating false positives.
6. **Robustness Analysis:** Compare results across different preprocessing settings and feature selections.
7. **Model Validation:** Use cross-validation and, where possible, external datasets to assess generalizability.

## 11. AI Disclosure

Generative AI tools, including ChatGPT, were used to assist with code development, debugging, and documentation.

The final implementation, analytical decisions, and reported findings were reviewed and validated by the project author.

## 12. Disclaimer

This project is intended for educational and exploratory data analysis purposes only.

It is not a validated clinical prediction system and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 13. Author

**Jinxiang Xiong**

Email: Xiongj18@msu.edu
