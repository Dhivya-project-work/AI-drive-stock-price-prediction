           # stock_eda_analysis.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("stock_data.csv")  # Make sure this file is in your working directory

# Select only numeric columns for analysis
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# 1. KDE Plots
plt.figure(figsize=(18, 12))
for i, column in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    sns.kdeplot(data=df[column].dropna(), fill=True, color='skyblue')
    plt.title(f'KDE Plot: {column}')
    plt.tight_layout()
plt.suptitle('Kernel Density Estimation (KDE)', y=1.02, fontsize=16)
plt.show()

# 2. Boxplots for Outlier Detection
plt.figure(figsize=(18, 12))
for i, column in enumerate(numeric_columns):
    plt.subplot(4, 4, i + 1)
    sns.boxplot(x=df[column].dropna(), color='orange')
    plt.title(f'Boxplot: {column}')
    plt.tight_layout()
plt.suptitle('Boxplots for Outlier Detection', y=1.02, fontsize=16)
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5)
plt.title('Correlation Heatmap', fontsize=16)
plt.show() 
