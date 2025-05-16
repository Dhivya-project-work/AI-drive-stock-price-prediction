import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

st.title("📈 Stock Data EDA Dashboard")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if not numeric_columns:
        st.warning("No numeric columns found in the dataset.")
    else:
        # KDE Plots
        st.subheader("📊 Kernel Density Estimation (KDE) Plots")
        num_cols = 4
        rows = (len(numeric_columns) + num_cols - 1) // num_cols
        for i in range(rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < len(numeric_columns):
                    with cols[j]:
                        fig, ax = plt.subplots()
                        sns.kdeplot(data=df[numeric_columns[idx]].dropna(), fill=True, color='skyblue', ax=ax)
                        ax.set_title(f'KDE Plot: {numeric_columns[idx]}')
                        st.pyplot(fig)

        # Boxplots
        st.subheader("🧪 Boxplots for Outlier Detection")
        for i in range(rows):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                idx = i * num_cols + j
                if idx < len(numeric_columns):
                    with cols[j]:
                        fig, ax = plt.subplots()
                        sns.boxplot(x=df[numeric_columns[idx]].dropna(), color='orange', ax=ax)
                        ax.set_title(f'Boxplot: {numeric_columns[idx]}')
                        st.pyplot(fig)

        # Correlation Heatmap
        st.subheader("📉 Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[numeric_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")
