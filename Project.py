# Student Performance Prediction - Final Project (Corrected Version)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans

from imblearn.over_sampling import SMOTE

# -------------------------
# Experiment 4: Dataset Details
# -------------------------
# Load dataset
df = pd.read_csv('student_data.csv')

# Basic info
print("Dataset Shape:", df.shape)
print("\nMemory Usage:\n", df.memory_usage())
print("\nMissing Values:\n", df.isnull().sum())
print("\nNumeric Summary:\n", df.describe())

# -------------------------
# Experiment 6: Missing Value & Outlier Handling
# -------------------------
# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

# Remove outliers using IQR
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# -------------------------
# Experiment 7: Feature Scaling & Encoding
# -------------------------
# One-hot encoding for categorical features
df = pd.get_dummies(df, drop_first=True)

# Feature scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('final_score', axis=1))
X = pd.DataFrame(scaled_features, columns=df.drop('final_score', axis=1).columns)
y = df['final_score']

# -------------------------
# Experiment 8: PCA
# -------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print("\nExplained Variance Ratio (2 components):", pca.explained_variance_ratio_)

# -------------------------
# Experiment 9: Handling Imbalanced Data
# -------------------------
# Create pass/fail target dynamically to avoid single-class issues
if df['final_score'].nunique() > 1:
    y_class = (y >= y.median()).astype(int)  # pass/fail threshold using median
    print("\nPass/Fail distribution:\n", y_class.value_counts())
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y_class)
else:
    print("\nAll students have the same final_score. Skipping SMOTE.")
    X_res, y_res = X, (y >= y.median()).astype(int)

# -------------------------
# Experiment 1 & 2: Regression & Classification
# -------------------------
# Classification
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("\nClassification Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Regression
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)
print("\nSample Regression Predictions:\n", y_pred_r[:5])

# -------------------------
# Experiment 3: Clustering
# -------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

plt.figure(figsize=(18, 12))  # bigger figure

# 1. Correlation Heatmap
plt.subplot(2, 2, 1)
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')

# 2. Final Score Histogram
plt.subplot(2, 2, 2)
sns.histplot(df['final_score'], bins=20, kde=True, color='skyblue')
plt.title('Final Score Distribution')

# 3. Pass/Fail Countplot
plt.subplot(2, 2, 3)
sns.countplot(x=y_class)
plt.xticks([0, 1], ['Fail', 'Pass'])
plt.title('Pass/Fail Count')

# 4. Boxplots for numeric features (limit to first 5 features)
plt.subplot(2, 2, 4)
numeric_cols_sample = numeric_cols[:5]  # limit number of features
df_melt = df.melt(id_vars='final_score', value_vars=numeric_cols_sample)
sns.boxplot(x='variable', y='value', data=df_melt)
plt.title('Boxplots of Numeric Features')
plt.xticks(rotation=45)

plt.tight_layout()  # automatically adjust spacing
plt.show()

# PCA Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y_class, palette='Set1')
plt.title('2D PCA Scatter Plot')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Optional: KMeans Clustering Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=df['Cluster'], palette='Set3')
plt.title('KMeans Clustering (PCA-reduced)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()
