import pandas as pd
import numpy as np

def load_data(data):
    # Load the data from the CSV file
    data = pd.read_excel(data)
    return data

# Function to remove outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


# Function to calculate RMSE
def calculate_rmse(true_vals, predicted_vals):
    return np.sqrt(np.mean((true_vals - predicted_vals) ** 2))