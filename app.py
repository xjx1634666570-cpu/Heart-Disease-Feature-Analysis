
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. Page Configuration
# ==========================================
st.set_page_config(
    page_title="Heart Disease Correlation Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Correlation Dashboard")
st.header("Heart Disease Risk Factor Analysis")
st.caption(
    "Explore the relationships between health indicators "
    "and heart disease status."
)

# ==========================================
# 2. Load Dataset
# ==========================================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)


uploaded_file = st.sidebar.file_uploader(
    "Upload Heart Disease Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data("heart_disease.csv")
    except FileNotFoundError:
        st.error(
            "heart_disease.csv was not found. "
            "Place the CSV file in the same directory as app.py "
            "or upload it using the sidebar."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.stop()

if df.empty:
    st.error("The dataset is empty. Please check your CSV file.")
    st.stop()

# ==========================================
# 3. Select Variables for Analysis
# ==========================================
target = "Heart Disease Status"

selected_cols = [
    "BMI",
    "Blood Pressure",
    "Homocysteine Level",
    "Triglyceride Level",
    "Alcohol Consumption",
    "Cholesterol Level",
    "Gender",
    "Smoking",
    "Age",
    "High Blood Pressure"
]

required_cols = selected_cols + [target]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    st.error(
        f"The following columns are missing: {missing_cols}\n\n"
        f"Available columns: {df.columns.tolist()}"
    )
    st.stop()

# Keep only the selected variables and target
df_selected = df[required_cols].copy()

# Remove leading and trailing whitespace from text values
for col in df_selected.select_dtypes(
    include=["object", "string"]
).columns:
    df_selected[col] = df_selected[col].str.strip()

# ==========================================
# 4. Data Cleaning and Encoding
# ==========================================

# Encode target variable
df_selected[target] = df_selected[target].replace({
    "No": 0,
    "Yes": 1
})

# Encode binary categorical variables
binary_cols = [
    "Gender",
    "Smoking",
    "Alcohol Consumption",
    "High Blood Pressure"
]

binary_mapping = {
    "Yes": 1,
    "No": 0,
    "Male": 1,
    "Female": 0,
    "High": 1,
    "Low": 0,
    "Medium": 0
}

for col in binary_cols:
    df_selected[col] = df_selected[col].replace(
        binary_mapping
    )

# Convert continuous variables to numeric
numeric_cols = [
    "BMI",
    "Blood Pressure",
    "Homocysteine Level",
    "Triglyceride Level",
    "Cholesterol Level",
    "Age"
]

for col in numeric_cols:
    df_selected[col] = pd.to_numeric(
        df_selected[col],
        errors="coerce"
    )

df_selected[target] = pd.to_numeric(
    df_selected[target],
    errors="coerce"
)

# Validate binary variables
for col in binary_cols:
    invalid_values = (
        df_selected[col].notna()
        & ~df_selected[col].isin([0, 1])
    )

    if invalid_values.any():
        examples = (
            df_selected.loc[invalid_values, col]
            .astype(str)
            .unique()[:5]
            .tolist()
        )

        st.error(
            f"Unrecognized values in '{col}': {examples}. "
            "Please check the original data encoding."
        )
        st.stop()

# Validate target variable
invalid_target = (
    df_selected[target].notna()
    & ~df_selected[target].isin([0, 1])
)

if invalid_target.any():
    st.error(
        "'Heart Disease Status' must contain Yes/No "
        "or 1/0 values."
    )
    st.stop()

# Remove rows with missing values
total_records = len(df_selected)
df_clean = df_selected.dropna().copy()
valid_records = len(df_clean)

if df_clean.empty:
    st.error(
        "No valid records remain after cleaning. "
        "Please check the dataset."
    )
    st.stop()

if df_clean[target].nunique() < 2:
    st.error(
        "The target variable must contain both classes "
        "to calculate correlations."
    )
    st.stop()

# ==========================================
# 5. Sidebar Settings
# ==========================================
st.sidebar.header("Analysis Settings")

st.sidebar.write(
    "Only the 10 selected variables are included."
)

show_data = st.sidebar.checkbox(
    "Show cleaned dataset",
    value=False
)

# ==========================================
# 6. Dataset Overview
# ==========================================
heart_cases = int((df_clean[target] == 1).sum())
no_heart_cases = int((df_clean[target] == 0).sum())

heart_rate = df_clean[target].mean() * 100

st.subheader("📊 Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{total_records:,}"
)

col2.metric(
    "Valid Records",
    f"{valid_records:,}",
    delta=f"{valid_records - total_records:+,}"
)

col3.metric(
    "Heart Disease Cases",
    f"{heart_cases:,}"
)

col4.metric(
    "Heart Disease Rate",
    f"{heart_rate:.2f}%"
)

st.divider()

# ==========================================
# 7. Correlation Analysis
# ==========================================
st.subheader("📈 Correlation Analysis")

analysis_df = df_clean[
    selected_cols + [target]
].copy()

# Calculate only correlations with the target variable
correlations = (
    analysis_df.corr(numeric_only=True)[target]
    .drop(labels=[target], errors="ignore")
    .dropna()
)

if correlations.empty:
    st.error(
        "Unable to calculate correlations. "
        "Please check the variable data types."
    )
    st.stop()

# Build correlation statistics table
corr_table = pd.DataFrame({
    "Variable": correlations.index,
    "Correlation": correlations.values
})

corr_table["Absolute Correlation"] = (
    corr_table["Correlation"].abs()
)

corr_table["Direction"] = np.where(
    corr_table["Correlation"] > 0,
    "Positive",
    "Negative"
)

corr_table = corr_table.sort_values(
    "Absolute Correlation",
    ascending=False
).reset_index(drop=True)

# ==========================================
# 8. Correlation Bar Chart
# ==========================================
st.markdown(
    "Red bars indicate positive correlations, while blue bars "
    "indicate negative correlations. Values closer to -1 or +1 "
    "represent stronger linear associations."
)

plot_data = corr_table.sort_values(
    "Correlation",
    ascending=True
)

colors = [
    "#E74C3C" if value > 0 else "#3498DB"
    for value in plot_data["Correlation"]
]

fig = go.Figure()

fig.add_trace(
    go.Bar(
        x=plot_data["Correlation"],
        y=plot_data["Variable"],
        orientation="h",
        marker_color=colors,
        text=[
            f"{value:.3f}"
            for value in plot_data["Correlation"]
        ],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Pearson r: %{x:.4f}"
            "<extra></extra>"
        )
    )
)

fig.add_vline(
    x=0,
    line_dash="dash",
    line_color="gray"
)

fig.update_layout(
    title="10 Selected Variables Correlated with Heart Disease",
    xaxis_title="Pearson Correlation Coefficient",
    yaxis_title="Variables",
    template="plotly_white",
    height=550,
    showlegend=False,
    margin=dict(l=30, r=60, t=70, b=50)
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# 9. Correlation Statistics Table
# ==========================================
st.subheader("📋 Correlation Statistics")

st.dataframe(
    corr_table.style.format({
        "Correlation": "{:.4f}",
        "Absolute Correlation": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

# Download correlation results
csv_data = corr_table.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="⬇️ Download Correlation Results",
    data=csv_data,
    file_name="heart_disease_correlations.csv",
    mime="text/csv"
)

# ==========================================
# 10. Heart Disease Status Distribution
# ==========================================
st.divider()
st.subheader("🥧 Heart Disease Status Distribution")

status_df = pd.DataFrame({
    "Status": [
        "No Heart Disease",
        "Heart Disease"
    ],
    "Count": [
        no_heart_cases,
        heart_cases
    ]
})

fig_pie = px.pie(
    status_df,
    names="Status",
    values="Count",
    hole=0.45,
    title="Heart Disease Status Distribution",
    color="Status",
    color_discrete_map={
        "No Heart Disease": "#3498DB",
        "Heart Disease": "#E74C3C"
    }
)

fig_pie.update_traces(
    textinfo="label+percent",
    hovertemplate=(
        "%{label}<br>"
        "Count: %{value}<br>"
        "Percentage: %{percent}"
        "<extra></extra>"
    )
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

# ==========================================
# 11. Correlation Heatmap
# ==========================================
st.divider()
st.subheader("🌡️ Correlation Heatmap")

heatmap_cols = selected_cols + [target]

heatmap_data = analysis_df[
    heatmap_cols
].corr(numeric_only=True)

fig_heatmap = px.imshow(
    heatmap_data,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    title="Correlation Matrix of Selected Variables"
)

fig_heatmap.update_layout(
    height=650,
    template="plotly_white"
)

st.plotly_chart(
    fig_heatmap,
    use_container_width=True
)

# ==========================================
# 12. Display Cleaned Dataset
# ==========================================
if show_data:
    st.divider()
    st.subheader("🔍 Cleaned Dataset")

    st.dataframe(
        df_clean,
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# 13. Footer
# ==========================================
st.divider()

st.caption(
    "Note: Pearson correlation measures linear association, "
    "not causation. Results depend on data quality, variable "
    "encoding, and missing-value handling. This dashboard is "
    "intended for exploratory analysis, not medical diagnosis."
)
