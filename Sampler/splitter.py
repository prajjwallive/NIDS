import pandas as pd

# Read the Excel file
file_path = 'sampled_data.csv'
df = pd.read_csv(file_path)

# Check if the number of rows is correct
if len(df) != 22544:
    raise ValueError("The number of rows in the Excel file is not 22544.")

# Define the split sizes
split_sizes = [1000, 5000, 10000, 15000]

# Shuffle the data before splitting
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Initialize the start index
start_idx = 0

# Split and save the data
for i, size in enumerate(split_sizes, 1):
    end_idx = start_idx + size
    sample_df = df_shuffled.iloc[start_idx:end_idx]
    sample_df.to_csv(f'sample_{i}.csv', index=False, header=True)
    start_idx = end_idx

# Save the remaining data
remaining_sample = df_shuffled.iloc[start_idx:]
remaining_sample.to_csv('sample_4.csv', index=False, header=True)

print("Data has been split and saved into CSV files.")
