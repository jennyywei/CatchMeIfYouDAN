import pandas as pd

def append_random_lines(source_file, target_file, n):
    source_df = pd.read_parquet(source_file)
    random_rows = source_df.sample(n=n)
    target_df = pd.read_parquet(target_file)
    updated_target_df = pd.concat([target_df, random_rows], ignore_index=True)
    updated_target_df.to_parquet(target_file)

# synthetic generation for train set
source_file_path = "datasets/pi_hackaprompt/train.parquet"
target_file_path = "datasets/pi_synthetic/train.parquet"
number_of_random_lines = 1000

# # synthetic generation for valid set
# source_file_path = "datasets/pi_hackaprompt/validation.parquet"
# target_file_path = "datasets/pi_synthetic/validation.parquet"
# number_of_random_lines = 200

append_random_lines(source_file_path, target_file_path, number_of_random_lines)