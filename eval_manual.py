import pandas as pd

def calculate_accuracy(csv_path, parquet_path):
    # Load the data from CSV and Parquet files
    csv_data = pd.read_csv(csv_path)
    parquet_data = pd.read_parquet(parquet_path)
    
    # Ensure the data is sorted by index if not already
    csv_data.sort_index(inplace=True)
    parquet_data.sort_index(inplace=True)
    
    # Determine the shortest length to avoid index out-of-range
    min_length = min(len(csv_data), len(parquet_data))
    
    # Truncate data to the shortest length
    csv_data = csv_data.iloc[:min_length]
    parquet_data = parquet_data.iloc[:min_length]
    
    # Compare labels
    matches = (csv_data['prediction'] == parquet_data['label'])
    
    # Calculate accuracy: number of matches divided by total comparisons
    accuracy = matches.mean()
    
    return accuracy

# Example usage
csv_path = 'results/recollection/pi_hackaprompt.csv'
parquet_path = 'datasets/pi_hackaprompt/test.parquet'
accuracy = calculate_accuracy(csv_path, parquet_path)
print(f'Accuracy: {accuracy:.2%}')