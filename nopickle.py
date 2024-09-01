import pandas as pd

# Define the path to the original dataset and the output CSV file
input_file = 'Train_data.csv'
output_file = 'selected_data.csv'

# List of selected columns to extract from the original dataset
selected_columns = [
    'protocol_type', 
    'service', 
    'flag', 
    'src_bytes', 
    'dst_bytes', 
    'count', 
    'same_srv_rate', 
    'diff_srv_rate', 
    'dst_host_srv_count', 
    'dst_host_same_srv_rate', 
]

try:
    # Read the original dataset into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Convert non-numeric data to numeric (floats)
    for col in selected_columns:
        if df[col].dtype == object:  # Check if the column contains non-numeric data
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Save the selected data to a new CSV file
    df[selected_columns].to_csv(output_file, index=False)

    print(f"Selected data saved to {output_file}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
