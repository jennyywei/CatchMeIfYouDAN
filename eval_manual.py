import pandas as pd

def calculate_accuracy(csv_path, parquet_path):
    csv_data = pd.read_csv(csv_path)
    parquet_data = pd.read_parquet(parquet_path)
    
    csv_data.sort_index(inplace=True)
    parquet_data.sort_index(inplace=True)
    
    min_length = min(len(csv_data), len(parquet_data))
    
    csv_data = csv_data.iloc[:min_length]
    parquet_data = parquet_data.iloc[:min_length]
    
    matches = (csv_data['prediction'] == parquet_data['label'])
    
    accuracy = matches.mean()
    
    return accuracy

csv_path = 'results/recollection/pi_hackaprompt.csv'
parquet_path = 'datasets/pi_hackaprompt/test.parquet'
accuracy = calculate_accuracy(csv_path, parquet_path)
print(f'Accuracy: {accuracy:.2%}')
