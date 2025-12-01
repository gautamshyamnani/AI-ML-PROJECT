Student Performance Analysis
Student Performance Prediction – Machine Learning Project
📌 Project Overview

This project aims to analyze and predict student performance based on various academic and personal factors.
Using Machine Learning techniques, the code predicts:

✔ Whether a student will Pass or Fail (Classification)
✔ The expected Final Score (Regression)
✔ Groups of similar students (Clustering)

Additionally, the project includes complete data preprocessing, visualization, and dimensionality reduction.

🧱 Dataset Details

The dataset used: student_data.csv
Features include:

Category	Example Features
Academic	Study hours, Attendance, Previous score
Personal	Gender, Parent education
Technology Access	Internet access
Target	Final Score
🔍 Key Techniques Used
Step	Description
Data Cleaning	Missing values imputed using Mean
Outlier Handling	Using IQR method
Feature Encoding	One-Hot Encoding for categorical variables
Feature Scaling	Using StandardScaler
Dimensionality Reduction	PCA (2 components)
Class Balancing	SMOTE applied for Pass/Fail prediction
ML Models	RandomForest Classifier & Regressor
Unsupervised Learning	KMeans clustering
Visualizations	Heatmap, Histogram, Boxplot, PCA plots
🧠 Machine Learning Models
1️⃣ Classification – Pass/Fail Prediction

Random Forest Classifier

Performance evaluated using:

Accuracy

Precision, Recall & F1 Score (classification_report)

2️⃣ Regression – Final Score Prediction

Random Forest Regressor

Predicted continuous final scores for students

3️⃣ Clustering – Finding Student Groups

KMeans Clustering

Students grouped into 3 performance clusters

📊 Visual Output

The project generates:

Correlation Heatmap

Final Score Distribution Histogram

Pass/Fail Count Plot

Boxplots for numeric features

PCA-based scatter plots

Clustering visualization

These help in understanding the data patterns clearly.

🛠 Requirements

Install the following Python libraries before running the code:

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

▶ How to Run

Place the dataset file: student_data.csv in the same folder as the script

Run the code in any Python environment:

python student_performance_prediction.py


View results in the console and generated graphs

🎯 Project Objectives Achieved

Understand dataset patterns affecting performance

Improve data quality using preprocessing steps

Predict academic success with high accuracy

Group students by performance similarity for better academic interventions

📌 Conclusion

The project successfully demonstrates how machine learning can be used to analyze and predict student outcomes. Schools and educators can use these insights to support students who may require academic assistance.
