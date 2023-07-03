import pandas as pd
import random

# Load the dataset from Excel file
df = pd.read_excel('./load_dataset/loadDataset.xlsx')

# Initialize a set to keep track of used product names
used_product_names = set()

# Function to generate random data for a product
def generate_product_data():
    available_products = set(df['Product Name']) - used_product_names
    if not available_products:
        raise ValueError("All product names have been used")

    product = random.choice(list(available_products))
    used_product_names.add(product)

    org_num = random.randint(1, 10)  # Change the range as needed
    org = f"inventory_management_{org_num}"
    cost = random.randint(50, 2000)
    profit = random.randint(20, 500)
    time = random.randint(30, 50)
    # Manpower calculation based on a constant factor
    manpower = random.randint(50, 250)

    return product, org, cost, profit, manpower , time

# Generate random data for a specified number of products
def generate_random_data(num_products):
    data = []
    for _ in range(num_products):
        try:
            product_data = generate_product_data()
            data.append(product_data)
        except ValueError as e:
            print(e)
            break
    return data

# Usage example
num_products_to_generate = 10
random_data = generate_random_data(num_products_to_generate)

# Create a DataFrame from the generated data
columns = ['Product Name', 'Organisation Name', 'cost to company','time', 'profit', 'manpower']
df_random_data = pd.DataFrame(random_data, columns=columns)

# Save the DataFrame to a CSV file
output_path = './dataset/product_data.csv'
df_random_data.to_csv(output_path, index=False)

print(f"Generated data saved to {output_path}")
