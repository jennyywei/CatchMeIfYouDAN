from pprint import pprint
import requests
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

from spotlighting import spotlighting
from eval import evaluate_output

############################## GLOBAL VARIABLES AND DEFS ##############################

API_KEY = "2446f4aba26f829a8e1238df75c078d7adb237fac3b6b077ac82a940d990bca8"
API_ENDPOINT = "https://api.together.xyz/v1/completions"
MODEL = "meta-llama/Llama-2-7b-chat-hf"
PROMPT_PATH = "prompts/prompt_os.txt"
WEIGHTS = [0.5, 0.5]
PATHS = {
    # "pi_class": [
    #     {"parquet": "datasets/pi_deepset/test.parquet", "csv_paths": ["danbert_model/test_preds/pi_deepset_danbert.csv", "dansn_model/test_preds/pi_deepset_dansn.csv"]}
    # ],
    "pi_detect": [
        {"parquet": "datasets/pi_hackaprompt/test.parquet", "csv_paths": ["danbert_model/test_preds/pi_hackaprompt_danbert.csv", "dansn_model/test_preds/pi_hackaprompt_dansn.csv"]}
    ]#,
    # "jailbreak": [
    #     {"parquet": "datasets/dan_jailbreak/test.parquet", "csv_paths": ["danbert_model/test_preds/dan_jailbreak_danbert.csv", "dansn_model/test_preds/dan_jailbreak_dansn.csv"]}#,
    #     # {"parquet": "datasets/protectai_jailbreak/test.parquet", "csv_paths": ["danbert_model/test_preds/protectai_jailbreak_danbert.csv", "dansn_model/test_preds/protectai_jailbreak_dansn.csv"]}
    # ],
    # "password": [
    #     # {"parquet": "datasets/lakera_ignore/test.parquet", "csv_paths": ["danbert_model/test_preds/lakera_ignore_danbert.csv", "dansn_model/test_preds/lakera_ignore_dansn.csv"]},
    #     # {"parquet": "datasets/lakera_summ/test.parquet", "csv_paths": ["danbert_model/test_preds/lakera_summ_danbert.csv", "dansn_model/test_preds/lakera_summ_dansn.csv"]},
    #     # {"parquet": "datasets/lakera_mosscap/test.parquet", "csv_paths": ["danbert_model/test_preds/lakera_mosscap_danbert.csv", "dansn_model/test_preds/lakera_mosscap_dansn.csv"]},
    #     {"parquet": "datasets/tensortrust_extraction/test.parquet", "csv_paths": ["danbert_model/test_preds/tensortrust_extraction_danbert.csv", "dansn_model/test_preds/tensortrust_extraction_dansn.csv"]}
    # ]
}

dataset_cats = {
    "pi_class": ["pi_deepset"],
    "pi_detect": ["pi_hackaprompt"],
    "jailbreak": ["dan_jailbreak", "protectai_jailbreak"],
    "password": ["lakera_ignore", "lakera_summ", "lakera_mosscap", "tensortrust_extraction"]
}


############################## NON-NEURAL PIPELINE ##############################

METHODS = ['dm', 'ec']
ENC = "rot13"
MALICIOUS = 1

def load_prompt(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()

def get_model_response(input_text):
    payload = {
        "model": MODEL,
        "prompt": input_text,
        "max_tokens": 200,
        "stop": ["</s>"]
    }
        
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    time.sleep(1)
    return response.json()["choices"][0]["text"]

def nonneural(row):
    sys_prompt = row['sys_prompt']
    user_input = row['user_input']
    sys_prompt2 = None
    if "sys_prompt2" in row:
        sys_prompt2 = row["sys_prompt2"]
    label = row['label']
    password = None
    if "password" in row:
        password = row['password']

    prompt = load_prompt(PROMPT_PATH)
    sys_prompt += "\n" + prompt
    spotlighted_input = spotlighting(sys_prompt, user_input, METHODS, sys_prompt2=sys_prompt2, encoding_method=ENC)
    full_input = spotlighted_input + "\n\n Response: "
    output = get_model_response(full_input)

    # print(output)

    return evaluate_output(output, password)


############################## NEURAL PIPELINE ##############################

def load_dataset(parquet_path):
    return pd.read_parquet(parquet_path)

def load_expert_evaluations(csv_paths):
    evaluations = []
    for path in csv_paths:
        df = pd.read_csv(path, header=None, names=['probability'])
        df['probability'] = pd.to_numeric(df['probability'], errors='coerce') 
        evaluations.append(df)
    return pd.concat(evaluations, axis=1)

def average_expert_probability(expert_evaluations, weights):
    # weighted average across the concatenated columns (experts)
    # return np.average(expert_evaluations, axis=1, weights=weights)
    if expert_evaluations.shape[1] != len(weights):
        raise ValueError(f"Expected {len(weights)} expert evaluations, but got {expert_evaluations.shape[1]}")
    # Perform weighted average
    return np.average(expert_evaluations, axis=1, weights=weights)

def classify_entries(datasets, weights):
    results = {}
    timeout = 1800

    for category, dataset_info in datasets.items():
        category_results = []
        for dataset in tqdm(dataset_info, desc=f"Processing datasets for {category}"):
            start_time = time.time() 
            data = load_dataset(dataset['parquet'])
            expert_evaluations = load_expert_evaluations(dataset['csv_paths'])
            data['average_probability'] = average_expert_probability(expert_evaluations, weights)
            
            predictions = []
            for index, row in data.iterrows():
                if time.time() - start_time > timeout:
                    print(f"Timeout reached for {category}, moving to next dataset.")
                    break
                prediction = 'malicious' if row['average_probability'] >= 0.5 else nonneural(row)
                predictions.append(prediction)

            predictions_df = pd.DataFrame(predictions, columns=['prediction'])
            predictions_df.to_csv(f'results/recollection/pi_hackaprompt.csv', index=False)
            category_results.append(predictions_df)
        
        pprint(predictions)
        pprint(category_results)
        results[category] = category_results

    return results

def calculate_accuracy(data):
    total_entries = len(data)
    correct_predictions = (data['prediction'] == data['label']).sum()
    return correct_predictions / total_entries

def calculate_accuracies(classified_data):
    accuracies = {}
    for category, datasets in classified_data.items():
        category_accuracy = [calculate_accuracy(dataset) for dataset in datasets]
        accuracies[category] = {
            'average_accuracy': np.mean(category_accuracy),
            'individual_accuracies': category_accuracy
        }
    return accuracies


classified_data = classify_entries(PATHS, WEIGHTS)
accuracies = calculate_accuracies(classified_data)
pprint(accuracies)
