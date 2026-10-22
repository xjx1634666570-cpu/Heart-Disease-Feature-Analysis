import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Heart Disease Feature Analysis",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Risk Factor Analysis")
st.header("Linear Correlation vs. Model-Based Feature Importance")
st.caption(
    "Compare how variables rank under simple linear assumptions (Pearson) "
    "versus a non-linear machine learning model (Random Forest)."
)

# ==========================================
# 2. Load Dataset
# ==========================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

uploaded_file = st.sidebar.file_uploader("Upload Heart Disease Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data("heart_disease.csv")
    except FileNotFoundError:
        st.error("heart_disease.csv not found. Place it in the same directory or upload via sidebar.")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

# ==========================================
# 3. Sidebar Toggles (Variable Selection & Settings)
# ==========================================
st.sidebar.header("Data & Analysis Settings")

# Dynamic Switch: Whether to include the alcohol consumption variable
include_alcohol = st.sidebar.toggle(
    "Include 'Alcohol Consumption'", 
    value=False,
    help="Warning: Including this variable will drop ~26% of rows due to high missing values."
)

if include_alcohol:
    st.sidebar.warning("⚠️ Including Alcohol Consumption will reduce valid records to ~7,200.")
    selected_cols = [
        "BMI", "Blood Pressure", "Homocysteine Level", "Triglyceride Level",
        "Cholesterol Level", "Gender", "Smoking", "Age", "High Blood Pressure", 
        "Fasting Blood Sugar", "Alcohol Consumption"
    ]
    binary_cols = ["Gender", "Smoking", "High Blood Pressure", "Alcohol Consumption"]
else:
    st.sidebar.success("✅ Excluding Alcohol Consumption retains ~9,900 records.")
    selected_cols = [
        "BMI", "Blood Pressure", "Homocysteine Level", "Triglyceride Level",
        "Cholesterol Level", "Gender", "Smoking", "Age", "High Blood Pressure", 
        "Fasting Blood Sugar"
    ]
    binary_cols = ["Gender", "Smoking", "High Blood Pressure"]

use_binning = st.sidebar.toggle("Use Medical Binning (Ordinal)", value=True)
show_data = st.sidebar.checkbox("Show cleaned dataset", value=False)

# ==========================================
# 4. Select Variables & Data Cleaning
# ==========================================
target = "Heart Disease Status"
required_cols = selected_cols + [target]

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing columns in dataset: {missing_cols}")
    st.stop()

df_selected = df[required_cols].copy()

# Clean text
for col in df_selected.select_dtypes(include=["object", "string"]).columns:
    df_selected[col] = df_selected[col].str.strip()

# Encoding
df_selected[target] = df_selected[target].replace({"No": 0, "Yes": 1})
binary_mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0, "High": 1, "Low": 0, "Medium": 0}

for col in binary_cols:
    df_selected[col] = df_selected[col].replace(binary_mapping)

# Convert numeric
numeric_cols = [
    "BMI", "Blood Pressure", "Homocysteine Level", 
    "Triglyceride Level", "Cholesterol Level", "Age", "Fasting Blood Sugar"
]

for col in numeric_cols:
    df_selected[col] = pd.to_numeric(df_selected[col], errors="coerce")

df_selected[target] = pd.to_numeric(df_selected[target], errors="coerce")

# Complete Case Analysis
df_clean = df_selected.dropna().copy()

if df_clean.empty:
    st.error("No valid records remain after cleaning.")
    st.stop()

# ==========================================
# 5. Medical Binning (Optional)
# ==========================================
def apply_medical_binning(df):
    df_binned = df.copy()
    df_binned['BMI'] = pd.cut(df['BMI'], bins=[-np.inf, 18.5, 25, 30, 35, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
    df_binned['Blood Pressure'] = pd.cut(df['Blood Pressure'], bins=[-np.inf, 90, 120, 130, 140, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
    df_binned['Triglyceride Level'] = pd.cut(df['Triglyceride Level'], bins=[-np.inf, 100, 150, 200, 500, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
    df_binned['Fasting Blood Sugar'] = pd.cut(df['Fasting Blood Sugar'], bins=[-np.inf, 70, 100, 126, 200, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
    df_binned['Cholesterol Level'] = pd.cut(df['Cholesterol Level'], bins=[-np.inf, 150, 200, 240, 300, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
    df_binned['Homocysteine Level'] = pd.cut(df['Homocysteine Level'], bins=[-np.inf, 5, 10, 15, 30, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
    df_binned['Age'] = pd.cut(df['Age'], bins=[-np.inf, 30, 45, 60, 75, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)
    return df_binned

if use_binning:
    df_analysis = apply_medical_binning(df_clean)
    st.info("💡 **Medical Binning is ON:** Continuous variables converted to ordinal risk levels (1-5).")
else:
    df_analysis = df_clean.copy()
    st.warning("⚠️ **Raw Continuous Data is ON:** Using raw continuous values.")

# ==========================================
# 6. Dataset Overview
# ==========================================
heart_cases = int((df_analysis[target] == 1).sum())
no_heart_cases = int((df_analysis[target] == 0).sum())
heart_rate = df_analysis[target].mean() * 100

st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Valid Records", f"{len(df_analysis):,}")
col2.metric("Heart Disease Cases", f"{heart_cases:,}")
col3.metric("No Heart Disease", f"{no_heart_cases:,}")
col4.metric("Heart Disease Rate", f"{heart_rate:.2f}%")
st.divider()

# ==========================================
# 7. Calculations (Pearson & Random Forest)
# ==========================================
X = df_analysis[selected_cols].copy()
y = df_analysis[target].copy()

# 1. Pearson Correlation
correlations = X.corrwith(y, method='pearson').dropna()

# 2. Random Forest Feature Importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X, y)
importances = rf_model.feature_importances_
importance_series = pd.Series(importances, index=X.columns).dropna()

# Combine Results
results_df = pd.DataFrame({
    "Variable": correlations.index,
    "Pearson r": correlations.values,
    "Abs Pearson": correlations.abs().values,
    "Feature Importance": importance_series[correlations.index].values
}).sort_values("Feature Importance", ascending=False).reset_index(drop=True)

# ==========================================
# 8. Visualizations
# ==========================================
st.subheader("📈 1. Pearson Correlation (Linear Association)")
plot_pearson = results_df.sort_values("Pearson r", ascending=True)
colors_p = ["#E74C3C" if v > 0 else "#3498DB" for v in plot_pearson["Pearson r"]]

fig_pearson = go.Figure(go.Bar(
    x=plot_pearson["Pearson r"], y=plot_pearson["Variable"],
    orientation="h", marker_color=colors_p,
    text=[f"{v:.3f}" for v in plot_pearson["Pearson r"]], textposition="outside"
))
fig_pearson.add_vline(x=0, line_dash="dash", line_color="gray")
fig_pearson.update_layout(title="Pearson Correlation with Heart Disease", height=400, template="plotly_white", margin=dict(l=30, r=60, t=40, b=30))
st.plotly_chart(fig_pearson, use_container_width=True)

st.divider()

st.subheader("🌲 2. Random Forest Feature Importance (Non-Linear)")
st.markdown("""
**Why Random Forest Feature Importance?** 
Pearson correlation only captures linear relationships. In medicine, risk factors often have complex, non-linear interactions. 
**Feature Importance** from a Random Forest model measures how much each variable contributes to the model's predictive accuracy. *(Note: The model is trained with `class_weight='balanced'` to account for class imbalance).*
""")

plot_imp = results_df.sort_values("Feature Importance", ascending=True)

fig_imp = go.Figure(go.Bar(
    x=plot_imp["Feature Importance"], y=plot_imp["Variable"],
    orientation="h", marker_color="#27AE60",
    text=[f"{v:.3f}" for v in plot_imp["Feature Importance"]], textposition="outside"
))
fig_imp.update_layout(title="Random Forest Feature Importance", height=400, template="plotly_white", margin=dict(l=30, r=60, t=40, b=30))
st.plotly_chart(fig_imp, use_container_width=True)

# ==========================================
# 9. Summary Table
# ==========================================
st.subheader("📋 Combined Statistical Summary")
st.markdown("Compare how variables rank differently under linear correlation vs. machine learning importance.")
st.dataframe(
    results_df.style.format({"Pearson r": "{:.4f}", "Abs Pearson": "{:.4f}", "Feature Importance": "{:.4f}"}),
    use_container_width=True, hide_index=True
)

# Download Button
csv_data = results_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="⬇️ Download Combined Results",
    data=csv_data,
    file_name="heart_disease_feature_importance.csv",
    mime="text/csv"
)

# ==========================================
# 10. Dataset & Footer
# ==========================================
if show_data:
    st.divider()
    st.subheader("🔍 Processed Dataset")
    st.dataframe(df_analysis, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Note: Feature importance indicates how useful a variable was to the model, "
    "not a causal relationship. Alcohol Consumption has a high missing rate; "
    "toggling it on/off demonstrates the trade-off between variable coverage and dataset size."
)