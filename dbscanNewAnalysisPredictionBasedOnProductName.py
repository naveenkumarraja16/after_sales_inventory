# only dbscan based on product name
import os
import pandas as pd
from datetime import date
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from datetime import date, timedelta

# Step 1: Data Collection
sales_data = pd.read_csv('dataset/createdData.csv')

# Step 2: Analyze Weekly Sales Data
sales_data['Date'] = pd.to_datetime(sales_data['Date'], format="%Y-%m-%d")
sales_data = sales_data.groupby(['Date', 'Product Name'])['Quantity'].sum().reset_index()

# Step 3: Analyze Weekly Sales Data
sales_data['Week'] = sales_data['Date'].dt.isocalendar().week
weekly_sales = sales_data.groupby(['Product Name', 'Week', 'Date'])['Quantity'].sum().reset_index()


# Step 4: Find the Demanded Products for the Last Two Weeks
current_date = date.today()
last_two_weeks = [(current_date - timedelta(weeks=1)).isocalendar()[1], current_date.isocalendar()[1]]
product_demand = weekly_sales.groupby(['Product Name', 'Week'])['Quantity'].sum().reset_index()
product_demand = product_demand[product_demand['Week'].isin(last_two_weeks)]
most_demanded_products = product_demand.groupby('Product Name')['Quantity'].sum().reset_index().sort_values(
    by='Quantity', ascending=False)

# Step 5: Calculate Increase and Decrease in Demand
last_week_demand = product_demand[product_demand['Week'] == last_two_weeks[0]]
current_week_demand = product_demand[product_demand['Week'] == last_two_weeks[1]]
demand_change = pd.merge(current_week_demand, last_week_demand, on='Product Name', how='left')
demand_change['Change'] = demand_change['Quantity_x'] - demand_change['Quantity_y']
demand_change['Change(%)'] = (demand_change['Change'] / demand_change['Quantity_y']) * 100

# Separate increase and decrease in demand
increase_demand = demand_change[demand_change['Change'] > 0]
decrease_demand = demand_change[demand_change['Change'] < 0]

# Step 6: Perform One-Hot Encoding
encoder = OneHotEncoder(sparse=False)
product_encoded = encoder.fit_transform(product_demand[['Product Name']])

# Step 7: Apply DBSCAN Algorithm for Clustering
dbscan = DBSCAN(eps=3, min_samples=2)
clustering_result = dbscan.fit_predict(product_encoded)
product_demand['Cluster'] = clustering_result

# Step 8: Export the Clusters to Excel
excel_folder = 'clustering_results'
os.makedirs(excel_folder, exist_ok=True)  # Create the clustering_results directory
excel_filename = f"{excel_folder}/product_clusters_{date.today()}.xlsx"
excel_writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
product_demand.to_excel(excel_writer, sheet_name='Product Clusters', index=False)
# excel_writer._save()

# print(f"Clusters exported to {excel_filename}")




# Step 15: Save and Close the Excel File
excel_writer._save()
excel_writer.close()


print(f"Exported most demanded products, line plots, and charts to {excel_filename}")




