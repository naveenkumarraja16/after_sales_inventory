# # Here's the first code that focuses on analyzing weekly sales data,
# #  finding the most demanded products, calculating the increase and decrease in demand,
# #  and exporting the results to an Excel file:



# # This code performs the following steps:

# # It collects the sales data from a CSV file.
# # It analyzes the weekly sales data and finds the most demanded products.
# # It calculates the increase and decrease in demand between the last two weeks.
# # It separates the products with increased demand and decreased demand.
# # It creates an Excel file to export the results.
# # It exports the increase and decrease in demand to separate sheets in the Excel file.
# # It defines a list of clustering algorithms (KMeans, GaussianMixture, DBSCAN, AgglomerativeClustering, Birch).
# # It iterates over the algorithms, applies each algorithm to the product demand data, and exports the clusters to separate sheets in the Excel file.
# # It saves and closes the Excel file.


























#based on quantity


import os
import pandas as pd
from datetime import date
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
import matplotlib.pyplot as plt

# Step 1: Data Collection
sales_data = pd.read_csv('dataset/createdData.csv')

# Step 2: Analyze Weekly Sales Data
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format="%Y-%m-%d")
sales_data = sales_data.groupby(['Date', 'Product Name'])['Quantity'].sum().reset_index()

# Step 3: Analyze Weekly Sales Data
sales_data['Week'] = sales_data['Date'].dt.isocalendar().week
weekly_sales = sales_data.groupby(['Product Name', 'Week', 'Date'])['Quantity'].sum().reset_index()

# Step 4: Find the Demanded Products for the Last Two Weeks
last_two_weeks = weekly_sales['Week'].unique()[-3:]
product_demand = weekly_sales.groupby(['Product Name', 'Week'])['Quantity'].sum().reset_index()
product_demand = product_demand[product_demand['Week'].isin(last_two_weeks)]
most_demanded_products = product_demand.groupby('Product Name')['Quantity'].sum().reset_index().sort_values(
    by='Quantity', ascending=False)

# # Step 5: Calculate Increase and Decrease in Demand
last_week_demand = product_demand[product_demand['Week'] == last_two_weeks[0]]
current_week_demand = product_demand[product_demand['Week'] == last_two_weeks[1]]
# demand_change = pd.merge(current_week_demand, last_week_demand, on='Product Name', how='left')
# demand_change['Change'] = demand_change['Quantity_x'] - demand_change['Quantity_y']
# demand_change['Change(%)'] = (demand_change['Change'] / demand_change['Quantity_y']) * 100

# # Separate increase and decrease in demand
# increase_demand = demand_change[demand_change['Change'] > 0]
# decrease_demand = demand_change[demand_change['Change'] < 0]

# Step 8: Create a Pandas Excel Writer
excel_folder = 'clustering_results'
os.makedirs(excel_folder, exist_ok=True)  # Create the demanded_products directory
excel_filename = f"{excel_folder}/product_clustering_results_{date.today()}.xlsx"
excel_writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')

# # Step 9: Export Increase and Decrease in Demand to Excel
# increase_demand.to_excel(excel_writer, sheet_name='Increase in Demand', index=False)
# decrease_demand.to_excel(excel_writer, sheet_name='Decrease in Demand', index=False)

# Step 10: Define the list of algorithms
algorithms = [
    KMeans(n_clusters=3, random_state=42),
    GaussianMixture(n_components=3, random_state=42),
    DBSCAN(eps=3, min_samples=2),
    AgglomerativeClustering(n_clusters=3),
    Birch(n_clusters=3)
]

algorithm_names = {
    0: 'KMeans',
    1: 'GaussianMix',
    2: 'DBSCAN',
    3: 'AggClustering',
    4: 'Birch'
}

# Step 11: Export Most Demanded Products to Excel for each algorithm
for algorithm_index, algorithm in enumerate(algorithms):
    algorithm_name = algorithm_names[algorithm_index]
    
    if algorithm_name == 'DBSCAN':
        # Use DBSCAN for anomaly detection
        clustering_result = algorithm.fit_predict(product_demand[['Quantity']])
        product_demand['Cluster'] = clustering_result
        anomalies = product_demand[clustering_result == -1]
        anomalies.to_excel(excel_writer, sheet_name=f'{algorithm_name} Anomalies', index=False)
    # else:
        # Use other clustering algorithms for prediction
    algorithm.fit(product_demand[['Quantity']])
    if algorithm_name == 'GaussianMix':
        product_demand['Cluster'] = algorithm.predict(product_demand[['Quantity']])
    else:
        product_demand['Cluster'] = algorithm.labels_
    clusters = product_demand.groupby('Cluster')['Quantity'].mean().reset_index()
    clusters.to_excel(excel_writer, sheet_name=f'{algorithm_name} Clusters', index=False)

# Step 12: Save and Close the Excel File
excel_writer._save()
excel_writer.close()

print(f"Clustering Results, clusters to {excel_filename}")





























































# import os
# import pandas as pd
# from datetime import date
# from sklearn.cluster import KMeans
# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import DBSCAN
# from sklearn.cluster import AgglomerativeClustering
# from sklearn.cluster import Birch
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt

# # Step 1: Data Collection
# sales_data = pd.read_csv('developed_data/createdData.csv')

# # Step 2: Analyze Weekly Sales Data
# sales_data['Date'] = pd.to_datetime(sales_data['Date'], format="%d-%m-%Y")
# sales_data = sales_data.groupby(['Date', 'Product Name'])['Quantity'].sum().reset_index()

# # Step 3: Analyze Weekly Sales Data
# sales_data['Week'] = sales_data['Date'].dt.isocalendar().week
# weekly_sales = sales_data.groupby(['Product Name', 'Week', 'Date'])['Quantity'].sum().reset_index()

# # Step 4: Find the Demanded Products for the Last Two Weeks
# last_two_weeks = weekly_sales['Week'].unique()[-2:]
# product_demand = weekly_sales.groupby(['Product Name', 'Week'])['Quantity'].sum().reset_index()
# product_demand = product_demand[product_demand['Week'].isin(last_two_weeks)]
# most_demanded_products = product_demand.groupby('Product Name')['Quantity'].sum().reset_index().sort_values(
#     by='Quantity', ascending=False)

# # Step 5: Calculate Increase and Decrease in Demand
# last_week_demand = product_demand[product_demand['Week'] == last_two_weeks[0]]
# current_week_demand = product_demand[product_demand['Week'] == last_two_weeks[1]]
# demand_change = pd.merge(current_week_demand, last_week_demand, on='Product Name', how='left')
# demand_change['Change'] = demand_change['Quantity_x'] - demand_change['Quantity_y']
# demand_change['Change(%)'] = (demand_change['Change'] / demand_change['Quantity_y']) * 100

# # Separate increase and decrease in demand
# increase_demand = demand_change[demand_change['Change'] > 0]
# decrease_demand = demand_change[demand_change['Change'] < 0]

# # Step 8: Create a Pandas Excel Writer
# excel_folder = 'demanded_products'
# os.makedirs(excel_folder, exist_ok=True)  # Create the demanded_products directory
# excel_filename = f"{excel_folder}/demanded_products_based_on_products{date.today()}.xlsx"
# excel_writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')

# # Step 9: Export Increase and Decrease in Demand to Excel
# increase_demand.to_excel(excel_writer, sheet_name='Increase in Demand', index=False)
# decrease_demand.to_excel(excel_writer, sheet_name='Decrease in Demand', index=False)

# # Step 10: Define the list of algorithms
# algorithms = [
#     KMeans(n_clusters=3, random_state=42),
#     GaussianMixture(n_components=3, random_state=42),
#     DBSCAN(eps=3, min_samples=2),
#     AgglomerativeClustering(n_clusters=3),
#     Birch(n_clusters=3)
# ]

# algorithm_names = {
#     0: 'KMeans',
#     1: 'GaussianMix',
#     2: 'DBSCAN',
#     3: 'AggClustering',
#     4: 'Birch'
# }

# # Step 11: Export Most Demanded Products to Excel for each algorithm
# for algorithm_index, algorithm in enumerate(algorithms):
#     algorithm_name = algorithm_names[algorithm_index]
    
#     if algorithm_name == 'DBSCAN':
#         # Use DBSCAN for anomaly detection
#         clustering_result = algorithm.fit_predict(product_demand[['Quantity']])
#         product_demand['Cluster'] = clustering_result
#         anomalies = product_demand[clustering_result == -1]
#         anomalies.to_excel(excel_writer, sheet_name=f'{algorithm_name} Anomalies', index=False)
#     else:
#         # Use other clustering algorithms for prediction
#         algorithm.fit(product_demand[['Quantity']])
#         if algorithm_name == 'GaussianMix':
#             product_demand['Cluster'] = algorithm.predict(product_demand[['Quantity']])
#         else:
#             product_demand['Cluster'] = algorithm.labels_
#         clusters = product_demand.groupby('Cluster')['Quantity'].mean().reset_index()
#         clusters.to_excel(excel_writer, sheet_name=f'{algorithm_name} Clusters', index=False)

# # Step 12: Perform Prediction using Clusters
# X = product_demand[['Quantity']]
# y = product_demand['Cluster']

# # Example 1: Linear Regression for Prediction
# linear_regression_model = LinearRegression()
# linear_regression_model.fit(X, y)
# linear_regression_predictions = linear_regression_model.predict(X)
# linear_regression_predictions_df = pd.DataFrame({'Quantity': X['Quantity'], 'Cluster Prediction': linear_regression_predictions})
# linear_regression_predictions_df.to_excel(excel_writer, sheet_name='Linear Regression Predictions', index=False)

# # Example 2: Support Vector Machine (SVM) for Prediction
# svm_model = SVC()
# svm_model.fit(X, y)
# svm_predictions = svm_model.predict(X)
# svm_predictions_df = pd.DataFrame({'Quantity': X['Quantity'], 'Cluster Prediction': svm_predictions})
# svm_predictions_df.to_excel(excel_writer, sheet_name='SVM Predictions', index=False)

# # Example 3: Random Forest Classifier for Prediction
# random_forest_model = RandomForestClassifier()
# random_forest_model.fit(X, y)
# random_forest_predictions = random_forest_model.predict(X)
# random_forest_predictions_df = pd.DataFrame({'Quantity': X['Quantity'], 'Cluster Prediction': random_forest_predictions})
# random_forest_predictions_df.to_excel(excel_writer, sheet_name='Random Forest Predictions', index=False)

# # Step 13: Save and Close the Excel File
# excel_writer._save()
# excel_writer.close()

# print(f"Exported most demanded products, clusters, and predictions to {excel_filename}")


# Plotting the clusters and predictions
# plt.figure(figsize=(10, 6))
# plt.scatter(X, y, c=y, cmap='viridis')
# plt.scatter(X, linear_regression_predictions, marker='x', color='red', label='Linear Regression Prediction')
# plt.scatter(X, svm_predictions, marker='s', color='green', label='SVM Prediction')
# plt.scatter(X, random_forest_predictions, marker='d', color='blue', label='Random Forest Prediction')
# plt.xlabel('Quantity')
# plt.ylabel('Cluster')
# plt.title(f'{algorithm_name} Clustering and Predictions')
# plt.legend()
# plt.show()




