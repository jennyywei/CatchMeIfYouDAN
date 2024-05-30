from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from itertools import combinations
from pprint import pprint
import os
import json
import requests
import time
import pandas as pd

from eval import evaluate_output

############################## GLOBAL VARIABLES AND DEFS ##############################

API_KEY = "2446f4aba26f829a8e1238df75c078d7adb237fac3b6b077ac82a940d990bca8"
API_ENDPOINT = "https://api.together.xyz/v1/completions"
MODEL = "meta-llama/Llama-2-7b-chat-hf"

dataset_cats = {
    "pi_class": ["pi_deepset"],
    "pi_detect": ["pi_hackaprompt"],
    "jailbreak": ["dan_jailbreak", "protectai_jailbreak"],
    "password": ["lakera_ignore", "lakera_summ", "lakera_mosscap", "tensortrust_extraction"]
}

dataset_dir = "datasets"

############################## LOADING FILES AND TOOLS ##############################

# loads all dataset splits (train, valid, or test) for each dataset category
    # arg: split (str) - dataset split to load (train, validation, or test)
    # returns: dictionary containing the datasets (dfs) for each category
def load_split(split):
    if (split != "train" and split != "validation" and split != "test" and split != "validation_spotlighting"):
        print("Tried to load an invalid split")
        return
    
    datasets = {}
    for category, dataset_list in dataset_cats.items():
        datasets[category] = []
        for ds in dataset_list:
            path = os.path.join(dataset_dir, ds)
            file_path = os.path.join(path, f"{split}.parquet")
            if os.path.exists(file_path):
                datasets[category].append(pd.read_parquet(file_path))
            else:
                print(f"{ds} {split} split not found when loading datasets")
    return datasets

############################## RUN BASELINE ##############################

def get_baseline_output(sys_prompt, user_input, sys_prompt2=None):
    input_text = sys_prompt + "\n" + user_input
    if sys_prompt2 != None:
        input_text += "\n" + sys_prompt2

    payload = {
        "model": MODEL,
        "prompt": input_text + "\n\n Response: ",
        "max_tokens": 200,
        "stop": ["</s>"]
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    response = requests.post(API_ENDPOINT, json=payload, headers=headers)
    response = json.loads(response.text)
    output = response["choices"][0]["text"]

    # add a delay to avoid rate limiting
    time.sleep(1)
    return output


# evalutes the performance of the baseline model on the given datasets
    # args: 
    #     datasets (dict): dictionary of datasets for each category
    # results: none (saves to json file in results dir)
def evaluate_baseline(datasets, results_dir="results/baseline"):
    results = {}

    for category, dfs in datasets.items():
        if category == "pi_class":
            handle_classification_task_baseline(dfs, category, results)
                    
        elif category in ["pi_detect", "jailbreak"]:
            handle_detection_task_baseline(dfs, category, results)

        elif category == "password":
            handle_password_task_baseline(dfs, category, results)
        
    # save results as a json
    with open(os.path.join(results_dir, "baseline_metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Baseline results saved at results/baseline/baseline_metrics.json")


# helper function for evaluate_baseline that works on the pi_class datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    # returns: none - updates results without returning
def handle_classification_task_baseline(dfs, category, results):
    y_true = []
    y_pred = []
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            output = get_baseline_output(row["sys_prompt"], row["user_input"])
            y_true.append(row["label"])
            y_pred.append(1 if evaluate_output(output) else 0)
    
    stats = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    results[category] = {
        "stats": stats,
        "accuracy": accuracy,
        "cm": cm.tolist()
    }

    print("BASELINE", category)
    print("    ACCURACY: ", accuracy)
    print("    PREC, RECALL, F1: ", stats)
    print("    CONF MATRIX: ", cm)


# helper function for evaluate_baseline that works on the jailbreak and pi_detect datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    # returns: none - updates results without returning
def handle_detection_task_baseline(dfs, category, results):
    correct = 0
    total = 0
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            output = get_baseline_output(row["sys_prompt"], row["user_input"])
            if evaluate_output(output):
                correct += 1
            total += 1
    accuracy = correct / total
    results[category] = {"accuracy": accuracy}

    print("BASELINE", category)
    print("    ACCURACY: ", accuracy)


# helper function for evaluate_baseline that works on the password datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    # returns: none - updates results without returning
def handle_password_task_baseline(dfs, category, results):
    correct = 0
    total = 0
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            sys_prompt1 = row["sys_prompt1"] if "sys_prompt1" in row else row["sys_prompt"]
            sys_prompt2 = row["sys_prompt2"] if "sys_prompt2" in row else None
            user_input = row["user_input"]

            output = get_baseline_output(sys_prompt1, user_input, sys_prompt2=sys_prompt2)
            if evaluate_output(output, row["password"]):
                correct += 1
            total += 1
    accuracy = correct / total
    results[category] = {"accuracy": accuracy}

    print("BASELINE", category)
    print("    ACCURACY: ", accuracy)


############################## MAIN METHOD ##############################

def main():
    # train = load_split("train")
    # validation = load_split("validation")
    validation_spotlighting = load_split("validation_spotlighting")
    # test = load_split("test")
    print("Datasets loaded")

    evaluate_baseline(validation_spotlighting)

if __name__ == "__main__":
    main()