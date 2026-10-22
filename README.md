# Heart Disease Risk Factor Analysis Dashboard

## 1. Project Overview

This project explores the relationships between demographic, clinical, and lifestyle variables and heart disease status.

An interactive web application is developed using **Streamlit**, **Pandas**, **NumPy**, **Plotly**, and **Scikit-Learn**. The dashboard allows users to examine the dataset, visualize both linear correlations and non-linear machine learning feature importances, and dynamically explore how data transformation affects results.

The project addresses the following questions:

1. Which selected variables have the strongest **linear correlations** with heart disease status?
2. Why do simple linear correlations fail to capture medical reality, and how can **non-linear machine learning models** (Random Forest) reveal the true predictive power of variables?
3. How does **Feature Engineering (Medical Binning)** and handling missing data dynamically impact the analytical results?

## 2. Dataset

The dataset, `heart_disease.csv`, contains approximately 10,000 records (observation unit = 1 patient).

The analysis focuses on the following core variables:
- **Clinical & Demographics:** BMI, Blood Pressure, Homocysteine Level, Triglyceride Level, Cholesterol Level, Fasting Blood Sugar, Age, Gender
- **Lifestyle:** Smoking, Alcohol Consumption, High Blood Pressure

**Target Variable:** `Heart Disease Status`, with values of `Yes` and `No`.

*Note:* The dataset suffers from severe **class imbalance** (~80% No, ~20% Yes) and a high missing rate (~26%) in the `Alcohol Consumption` variable.

## 3. Technologies and Requirements

The project uses Python and the following libraries:

- **Streamlit:** Interactive web application framework
- **Pandas & NumPy:** Data loading and manipulation
- **Plotly:** Interactive charts and visualizations
- **Scikit-Learn:** Machine learning (Random Forest Classifier)

### Installation

Install the required dependencies:

```bash
pip install streamlit pandas numpy plotly scikit-learn
Alternatively, create a requirements.txt file containing:

text
streamlit
pandas
numpy
plotly
scikit-learn
Then install the dependencies with:

bash
pip install -r requirements.txt
4. How to Run the Application
Download or clone the project files.

Ensure that app.py and heart_disease.csv are in the same directory.

Open a terminal in the project directory.

Run the following command:

bash
streamlit run app.py
Open the application in a browser at http://localhost:8501. The application also supports uploading a CSV file through the sidebar.

5. Project Structure
text
HeartDiseaseProject/
├── app.py
├── heart_disease.csv
├── README.md
└── requirements.txt
6. Streamlit Dashboard Features
6.1 Dynamic Data Handling (Sidebar Toggles)
Include 'Alcohol Consumption': Because alcohol has a ~26% missing rate, users can toggle this variable on/off. Toggling it on reduces valid records to ~7,200; toggling it off retains ~9,900 records.

Medical Binning (Ordinal Variables): Converts continuous variables (Age, BMI, Blood Pressure, etc.) into medical risk levels (1 to 5) to capture non-linear threshold effects.

6.2 Dataset Overview
The dashboard displays dynamic summary metrics: Total Records, Valid Records, Heart Disease Cases, and Heart Disease Rate.

6.3 Linear Analysis (Pearson Correlation)
A horizontal bar chart displays the Pearson correlation coefficient between each selected variable and heart disease status.

Red represents positive correlation; Blue represents negative correlation.

6.4 Non-Linear Analysis (Random Forest Feature Importance)
A machine learning model (Random Forest) is trained using class_weight='balanced' to address the class imbalance.

A green bar chart visualizes the Feature Importance, revealing non-linear predictive power that Pearson correlation misses.

6.5 Heart Disease Status Distribution & Heatmap
A donut chart visualizes the proportion of positive and negative cases in the cleaned dataset.

A heatmap displays pairwise correlations among the selected variables to identify potential overlap.

6.6 Cleaned Dataset Preview
Users can enable the dataset preview option in the sidebar to inspect the processed records (including binned variables).

7. Data Preparation
The application performs the following preprocessing steps:

Selects the specified variables and the target variable.

Removes leading and trailing whitespace from text values.

Converts the target from Yes/No to 1/0, and encodes binary variables.

Converts continuous measurements to numeric types.

Applies Complete Case Analysis (removes rows with missing values) to avoid artificial bias from imputation.

Applies Medical Binning to continuous variables if toggled.

Trains a Random Forest model to extract feature importance scores.

8. Key Analytical Considerations
8.1 The Limitation of Linear Correlation
Pearson correlation measures only linear associations. In medical data, risk factors often have U-shaped curves or threshold effects (e.g., risk spikes only after BMI > 30). Thus, low Pearson scores (all < 0.05) do not mean variables are useless.

8.2 The Power of Machine Learning
Random Forest captures complex interactions. The model correctly identified Age as the dominant predictor, followed by BMI and Cholesterol, which aligns perfectly with clinical consensus.

8.3 Class Imbalance
Because 80% of records are "No", a naive model would achieve 80% accuracy by always guessing "No". This project applies class_weight='balanced' in the Random Forest and emphasizes that future modeling must prioritize Recall and F1-Score over Accuracy.

9. Limitations
Not a Diagnostic Tool: This is an exploratory analysis, not a validated clinical system.

No Causation: Correlation and feature importance do not establish causation.

Dataset Quality: The weak linear correlations and relatively low feature importance scores suggest high noise or synthetic elements in the dataset.

Missing Value Bias: Removing rows with missing data may affect representativeness.

10. Future Improvements
Implementing predictive modeling (e.g., Logistic Regression vs. Random Forest) to evaluate Recall and F1-score.

Investigating SHAP values for deeper machine learning interpretability.

Using imputation methods (like KNN Imputer) to retain the Alcohol Consumption variable without losing 26% of the dataset.

11. AI Disclosure
Generative AI tools, including ChatGPT, were used to assist with code development, debugging, and documentation. The final implementation, analytical decisions, and reported findings were reviewed and validated by the project author.

12. Disclaimer
This project is intended for educational and exploratory data analysis purposes only. It is not a validated clinical prediction system and must not be used as a substitute for professional medical advice, diagnosis, or treatment.

13. Author
Jinxiang Xiong
Email: Xiongj18@msu.edu

text
