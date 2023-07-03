import os
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the product_demand data
product_demand = pd.read_csv('dataset/createdData.csv')

# Step 2: Perform Clustering based on Quantity
X = product_demand[['Quantity']]
clustering_algorithm = KMeans(n_clusters=3, random_state=42)
clustering_algorithm.fit(X)
product_demand['Cluster'] = clustering_algorithm.labels_

# Step 3: Encode the Product Name
label_encoder = LabelEncoder()
product_demand['Product Name Encoded'] = label_encoder.fit_transform(product_demand['Product Name'])

# Step 4: Create a list of unique product names
unique_product_names = product_demand['Product Name'].unique()


# Step 5: Perform Prediction for each unique product name based on the last two weeks' sales data
predictions = pd.DataFrame(columns=['Product Name', 'Week Number', 'Date', 'Sales',
                                    'Linear Regression Prediction', 'SVM Prediction', 'Random Forest Prediction','Total Sales'])
for product_name in unique_product_names:
    product_data = product_demand[product_demand['Product Name'] == product_name]

    # Convert 'Date' column to datetime object
    product_data.loc[:, 'Date'] = pd.to_datetime(product_data['Date'])

    # Calculate the date range for the last two weeks
    end_date = product_data['Date'].max()
    start_date = end_date - timedelta(weeks=1)

    # Filter the data for the last two weeks
    last_two_weeks_data = product_data.loc[(product_data['Date'] >= start_date) & (product_data['Date'] <= end_date)]

    if len(last_two_weeks_data) > 0:
        X = last_two_weeks_data[['Quantity', 'Product Name Encoded']]
        y = last_two_weeks_data['Cluster']

        # Linear Regression for Prediction
        linear_regression_model = LinearRegression()
        linear_regression_model.fit(X, y)
        linear_regression_predictions = linear_regression_model.predict(X)

        # SVM for Prediction
        if len(y.unique()) > 1:
            svm_model = SVC()
            svm_model.fit(X, y)
            svm_predictions = svm_model.predict(X)
        else:
            continue

        # Random Forest Classifier for Prediction
        random_forest_model = RandomForestClassifier()
        random_forest_model.fit(X, y)
        random_forest_predictions = random_forest_model.predict(X)

        # Calculate the week number and total sales for the last two weeks
        week_number = end_date.isocalendar()[1]
        total_sales = last_two_weeks_data['Quantity'].sum()

        # Append predictions to the overall predictions dataframe
        for index, row in last_two_weeks_data.iterrows():
            predictions = predictions._append({
                'Product Name': product_name,
                'Week Number': row['Date'].isocalendar()[1],
                'Date': row['Date'],
                'Sales': row['Quantity'],
                'Linear Regression Prediction': linear_regression_predictions[index % len(linear_regression_predictions)],
                'SVM Prediction': svm_predictions[index % len(svm_predictions)],
                'Random Forest Prediction': random_forest_predictions[index % len(random_forest_predictions)],
                'Total Sales':total_sales,
            }, ignore_index=True)
        


# Step 6: Create a Pandas Excel Writer
excel_folder = 'prediction_results'
os.makedirs(excel_folder, exist_ok=True)  # Create the prediction_results directory
excel_filename = f"{excel_folder}/cluster_predictions_based_on_productName_{datetime.today().date()}.xlsx"
excel_writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')

# Step 7: Export Predictions to Excel
predictions.to_excel(excel_writer, sheet_name='Cluster Predictions', index=False)

# Step 8: Format the date column and week number column in the Excel sheet
worksheet = excel_writer.sheets['Cluster Predictions']
date_format = excel_writer.book.add_format({'num_format': 'yyyy-mm-dd'})
worksheet.set_column('C:C', None, date_format)
worksheet.set_column('B:B', None, None, {'header': 'Week Number'})

# Step 8: Save and Close the Excel File
excel_writer._save()
excel_writer.close()

print(f"Weekly predictions exported to {excel_filename}")
