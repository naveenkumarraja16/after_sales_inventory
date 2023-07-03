# This code performs the following steps:

# It collects the sales data from a CSV file.
# It analyzes the weekly sales data and finds the product demand for the last two weeks.
# It defines a list of prediction algorithms (Linear Regression, SVM, Random Forest).
# It iterates over the algorithms and performs prediction using the cluster labels as the target variable.
# It adds the cluster predictions to the product demand dataframe.
# It creates an Excel file to export the cluster predictions.
# It exports the product demand dataframe with cluster predictions to the Excel file.
# It saves and closes the Excel file.

# Here's the second code that focuses on performing predictions using clusters

import os
import pandas as pd
from datetime import date
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Step 1: Load the product_demand data
product_demand = pd.read_csv('dataset/createdData.csv')

# Step 2: Perform Clustering
X = product_demand[['Quantity']]

# Choose one of the clustering algorithms: KMeans, GaussianMixture, DBSCAN, AgglomerativeClustering, or Birch
clustering_algorithm = KMeans(n_clusters=3, random_state=42)
clustering_algorithm.fit(X)
product_demand['Cluster'] = clustering_algorithm.labels_

# Step 3: Perform Prediction using Clusters
X = product_demand[['Quantity']]
y = product_demand['Cluster']

# Example 1: Linear Regression for Prediction
linear_regression_model = LinearRegression()
linear_regression_model.fit(X, y)
linear_regression_predictions = linear_regression_model.predict(X)
linear_regression_predictions_df = pd.DataFrame({'Quantity': X['Quantity'], 'Cluster Prediction': linear_regression_predictions})

# Example 2: Support Vector Machine (SVM) for Prediction
svm_model = SVC()
svm_model.fit(X, y)
svm_predictions = svm_model.predict(X)
svm_predictions_df = pd.DataFrame({'Quantity': X['Quantity'], 'Cluster Prediction': svm_predictions})

# Example 3: Random Forest Classifier for Prediction
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X, y)
random_forest_predictions = random_forest_model.predict(X)
random_forest_predictions_df = pd.DataFrame({'Quantity': X['Quantity'], 'Cluster Prediction': random_forest_predictions})

# Step 4: Create a Pandas Excel Writer
excel_folder = 'prediction_results'
os.makedirs(excel_folder, exist_ok=True)  # Create the prediction_results directory
excel_filename = f"{excel_folder}/cluster_predictions_{date.today()}.xlsx"
excel_writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')

# Step 5: Export Predictions to Excel
linear_regression_predictions_df.to_excel(excel_writer, sheet_name='Linear Regression Predictions', index=False)
svm_predictions_df.to_excel(excel_writer, sheet_name='SVM Predictions', index=False)
random_forest_predictions_df.to_excel(excel_writer, sheet_name='Random Forest Predictions', index=False)

# Step 6: Save and Close the Excel File
excel_writer._save()
excel_writer.close()

print(f"Cluster predictions exported to {excel_filename}")
