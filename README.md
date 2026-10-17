# Heart Disease Correlation Analysis Dashboard

## 1. Project Overview

This project explores the relationships between demographic, clinical, and lifestyle variables and heart disease status.

An interactive web application is developed using **Streamlit**, **Pandas**, **NumPy**, and **Plotly**. The dashboard allows users to examine the dataset, visualize correlations, and explore the distribution of heart disease status.

The project addresses the following questions:

1. Which selected variables have the strongest linear correlations with heart disease status?
2. How are the selected health indicators correlated with one another?
3. How can interactive visualizations support exploratory data analysis?

## 2. Dataset

The dataset, `heart_disease.csv`, contains approximately 10,000 records.

The analysis focuses on the following 10 variables:

- BMI
- Blood Pressure
- Homocysteine Level
- Triglyceride Level
- Alcohol Consumption
- Cholesterol Level
- Gender
- Smoking
- Age
- High Blood Pressure

**Target Variable:** `Heart Disease Status`, with values of `Yes` and `No`.

Other variables in the original dataset are excluded from the current correlation analysis.

## 3. Technologies and Requirements

The project uses Python and the following libraries:

- **Streamlit:** Interactive web application framework
- **Pandas:** Data loading and manipulation
- **NumPy:** Numerical operations
- **Plotly:** Interactive charts and visualizations

### Installation

Install the required dependencies:

```bash
pip install streamlit pandas numpy plotly
```

Alternatively, create a `requirements.txt` file containing:

```text
streamlit
pandas
numpy
plotly
```

Then install the dependencies with:

```bash
pip install -r requirements.txt
```

## 4. How to Run the Application

1. Download or clone the project files.
2. Ensure that `app.py` and `heart_disease.csv` are in the same directory.
3. Open a terminal in the project directory.
4. Run the following command:

   ```bash
   streamlit run app.py
   ```

5. Open the application in a browser at:

   `http://localhost:8501`

The application also supports uploading a CSV file through the sidebar.

## 5. Project Structure

```text
HeartDiseaseProject/
├── app.py
├── heart_disease.csv
├── README.md
└── requirements.txt
```

## 6. Streamlit Dashboard Features

### 6.1 Dataset Overview

The dashboard displays four summary metrics:

- Total Records
- Valid Records
- Heart Disease Cases
- Heart Disease Rate

These metrics provide an overview of the dataset after the selected variables have been cleaned.

### 6.2 Correlation Analysis

The application calculates the Pearson correlation coefficient between each selected variable and heart disease status.

A horizontal bar chart displays the results:

- Red represents positive correlation.
- Blue represents negative correlation.
- Values closer to +1 or -1 indicate stronger linear associations.
- Values closer to 0 indicate weaker linear associations.

The chart supports interactive exploration through Plotly.

### 6.3 Correlation Statistics Table

A table presents each variable's:

- Pearson correlation coefficient
- Absolute correlation coefficient
- Correlation direction

Users can download the correlation results as a CSV file.

### 6.4 Heart Disease Status Distribution

A donut chart visualizes the distribution of records with and without heart disease.

This chart helps users understand the proportion of positive and negative cases in the cleaned dataset.

### 6.5 Correlation Heatmap

The heatmap displays pairwise Pearson correlations among the 10 selected variables and the target variable.

It helps identify potential relationships between health indicators and possible overlap among explanatory variables.

### 6.6 Cleaned Dataset Preview

Users can enable the dataset preview option in the sidebar to inspect the cleaned records used in the analysis.

## 7. Data Preparation

The application performs the following preprocessing steps:

1. Loads the CSV dataset.
2. Selects the 10 specified variables and the target variable.
3. Removes leading and trailing whitespace from text values.
4. Converts the heart disease target from `Yes`/`No` to `1`/`0`.
5. Encodes supported binary categorical variables.
6. Converts continuous measurements to numeric types.
7. Removes records with missing values in the selected columns.
8. Calculates Pearson correlations using the cleaned data.

The application excludes unrelated categorical variables rather than creating thousands of dummy variables.

## 8. Key Analytical Considerations

### 8.1 Correlation Does Not Imply Causation

Pearson correlation measures linear association. A positive correlation does not establish that a variable causes heart disease, and a negative correlation does not establish a protective effect.

### 8.2 Correlation Is Not a Complete Measure of Predictive Power

A variable with a weak Pearson correlation may still contribute to a predictive model through nonlinear relationships or interactions with other variables.

### 8.3 Class Imbalance

The proportion of positive and negative cases should be assessed using the actual dataset after cleaning.

If the dataset is imbalanced, future predictive modeling should evaluate metrics such as recall, precision, F1-score, and ROC-AUC rather than relying on accuracy alone.

### 8.4 Data Encoding

Categorical variables must be encoded according to their actual meanings. For example, ordinal categories should not automatically be treated as equivalent binary categories.

The encoding of alcohol consumption and blood pressure variables should be verified against the original dataset documentation.

### 8.5 Missing Data

The current implementation removes rows with missing values in the selected variables. This can reduce the sample size and may introduce bias if missingness is systematic.

## 9. Limitations

- The current dashboard focuses on correlation analysis rather than developing or validating a predictive model.
- Pearson correlation captures linear associations and may not reveal nonlinear relationships.
- Missing-value removal may affect the representativeness of the analyzed sample.
- The analysis does not establish causation or control for potential confounding variables.
- Findings depend on data quality, sample composition, and variable encoding.

## 10. Future Improvements

Potential extensions include:

- Adding interactive filters for age and gender.
- Exploring BMI and triglyceride relationships through scatter plots.
- Engineering combined risk-related features for exploratory analysis.
- Comparing logistic regression and nonlinear classification models.
- Evaluating model performance using recall, F1-score, and ROC-AUC.
- Investigating class imbalance using appropriate resampling methods.
- Adding model interpretation tools to explain predictive results.

These extensions should be implemented and evaluated before being described as completed application features.

## 11. AI Disclosure

Generative AI tools, including ChatGPT, were used to assist with code development, debugging, and documentation.

The final implementation, analytical decisions, and reported findings should be reviewed and validated by the project author.

## 12. Disclaimer

This project is intended for educational and exploratory data analysis purposes only. It is not a validated clinical prediction system and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

## 13. Author

**Jinxiang Xiong**

Email: Xiongj18@msu.edu