import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Define the path to your original dataset and the output CSV file
input_file = 'Test_data.csv'
output_file = 'sampled_data.csv'

# Define the selected features including categorical ones
selected_features = ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
                     'count', 'same_srv_rate', 'diff_srv_rate', 'dst_host_srv_count',
                     'dst_host_same_srv_rate']

try:
    # Read the original dataset into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Select only the columns of interest
    df = df[selected_features]

    # Perform label encoding for categorical features
    label_encoders = {}
    categorical_features = ['protocol_type', 'service', 'flag']
    
    for feature in categorical_features:
        label_encoders[feature] = LabelEncoder()
        df[feature] = label_encoders[feature].fit_transform(df[feature])

    # Save the sampled data to a new CSV file
    df.to_csv(output_file, index=False)

    print(f"Sampled data saved to {output_file}")

except FileNotFoundError:
    print(f"Error: The file {input_file} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
