from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from itertools import combinations
from pprint import pprint
import os
import json
import requests
import time
import pandas as pd

from spotlighting import spotlighting
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

prompt_paths = ["prompts/prompt_zs.txt", 
                "prompts/prompt_os.txt", 
                "prompts/prompt_fs.txt"]

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


# loads zero, one, and few shot prompts from the specified file paths
    # arg: prompt_paths (list) - list of file paths to the prompt files
    # returns: dictionary of prompts corresponding to keys zs, os, and fs
def load_prompts(prompt_paths):
    prompts = {}
    for name, file_path in zip(["zs", "os", "fs"], prompt_paths):
        with open(file_path, "r") as file:
            prompts[name] = file.read().strip()
    return prompts


# generates all possible combinations of spotlighting methods
    # no args
    # returns: list of tuples that represent a combination of spotlighting methods
def generate_spotlighting_combos():
    methods = ["dl", "dm", "ec"] # delimiting, datamarking, encoding
    combs = []
    for i in range(1, len(methods) + 1):
        combs.extend(combinations(methods, i))
        
    return combs

############################## RUN SPOTLIGHTING TESTS ##############################

# applies spotlighting techniques and generates output from given system prompt & user input
    # args: 
    #     sys_prompt (str) - system prompt
    #     user_input (str) - user input
    #     prompt_type (str) - type of the prompt (zs, os, fs)
    #     prompts (dict) - the dictionary of prompts
    #     methods (list) - a list of spotlighting methods to apply
    #     sys_prompt2 (str, optional) - optional system prompt after the user input
    # returns: the output generated (str)
def get_spotlighting_output(sys_prompt, user_input, prompt, methods, sys_prompt2=None):
    # encoding_method = "base64"
    encoding_method = "rot13" # keeps text length the same
    # encoding_method = "binary"

    sys_prompt += "\n" + prompt
    spotlighted_input = spotlighting(sys_prompt, user_input, methods, sys_prompt2, encoding_method=encoding_method)

    payload = {
        "model": MODEL,
        "prompt": spotlighted_input + "\n\n Response: ",
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


# evalutes the performance of spotlighting (and prompting, which is secondary) on the given datasets
    # args: 
    #     datasets (dict): dictionary of datasets for each category
    # results: none (saves to json file in results dir)
def evaluate_spotlighting(datasets, results_dir="results/spotlighting"):
    results = {}
    os_prompt = load_prompts(prompt_paths)["os"] # most successful kind of prompting from baseline
    spotlighting_methods = generate_spotlighting_combos()

    for methods in tqdm(spotlighting_methods, total=len(spotlighting_methods), desc="Spotlighting Methods"): # and all combinations of spotlighting methods
        spotlighting_type = "_".join(methods)
        results[spotlighting_type] = {}

        for category, dfs in datasets.items():
            # all pi_class datasets are classification tasks, so we analyze all true and mis classifications
            # store precision, accuracy, recall, f1, and confusion matrix metrics
            if category == "pi_class":
                handle_classification_task(dfs, category, results[spotlighting_type], os_prompt, methods)
                
            # all pi_detect and jailbreak datasets are malicious, so we only evaluate accuracy based on if malicious input is detected
            elif category in ["pi_detect", "jailbreak"]:
                handle_detection_task(dfs, category, results[spotlighting_type], os_prompt, methods)

            # all password datasets are malicious, so we only evaluate accuracy based on whether the password is present
            elif category == "password":
                handle_password_task(dfs, category, results[spotlighting_type], os_prompt, methods)
    
        # save results as a json
        with open(os.path.join(results_dir, f"{spotlighting_type}_metrics.json"), "w") as f:
            json.dump(results[spotlighting_type], f, indent=4)
        print(f"Results saved at results/spotlighting/{spotlighting_type}_metrics.json")
    

# helper function for evaluate_spotlighting_prompting that works on the pi_class datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    #     prompt_type (str) - type of prompting (zs, os, fs)
    #     prompts (list) - list of loaded prompts
    #     methods (list) - list of spotlighting methods
    # returns: none - updates results without returning
def handle_classification_task(dfs, category, results, prompt, methods):
    y_true = []
    y_pred = []
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            try:
                output = get_spotlighting_output(row["sys_prompt"], row["user_input"], prompt, methods)
                y_true.append(row["label"])
                y_pred.append(1 if evaluate_output(output) else 0)
            except Exception as e:
                print(f"Error processing data point: {e}")
    
    stats = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    results[category] = {
        "stats": stats,
        "accuracy": accuracy,
        "cm": cm.tolist()
    }

    print("TECHNIQUES: ", f"{"_".join(methods)}", category)
    print("    ACCURACY: ", accuracy)
    print("    PREC, RECALL, F1: ", stats)
    print("    CONF MATRIX: ", cm)


# helper function for evaluate_spotlighting_prompting that works on the jailbreak and pi_detect datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    #     prompt_type (str) - type of prompting (zs, os, fs)
    #     prompts (list) - list of loaded prompts
    #     methods (list) - list of spotlighting methods
    # returns: none - updates results without returning
def handle_detection_task(dfs, category, results, prompt, methods):
    correct = 0
    total = 0
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            try:
                output = get_spotlighting_output(row["sys_prompt"], row["user_input"], prompt, methods)
                if evaluate_output(output):
                    correct += 1
                total += 1
            except Exception as e:
                print(f"Error processing data point: {e}")
    accuracy = correct / total
    results[category] = {"accuracy": accuracy}

    print("TECHNIQUES: ", f"{"_".join(methods)}", category)
    print("    ACCURACY: ", accuracy)


# helper function for evaluate_spotlighting_prompting that works on the password datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    #     prompt_type (str) - type of prompting (zs, os, fs)
    #     prompts (list) - list of loaded prompts
    #     methods (list) - list of spotlighting methods
    # returns: none - updates results without returning
def handle_password_task(dfs, category, results, prompt, methods):
    correct = 0
    total = 0
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            try:
                # lakera datasets only have 1 sys_prompt, but tensortrust dataset has 2
                sys_prompt1 = row["sys_prompt1"] if "sys_prompt1" in row else row["sys_prompt"]
                sys_prompt2 = row["sys_prompt2"] if "sys_prompt2" in row else None
                output = get_spotlighting_output(sys_prompt1, row["user_input"], prompt, methods, sys_prompt2=sys_prompt2)
                if evaluate_output(output, row["password"]):
                    correct += 1
                total += 1
            except Exception as e:
                print(f"Error processing data point: {e}")
    accuracy = correct / total
    results[category] = {"accuracy": accuracy}

    print("TECHNIQUES: ", f"{"_".join(methods)}", category)
    print("    ACCURACY: ", accuracy)


############################## RUN PROMPTING TESTS ##############################

def get_prompting_output(sys_prompt, user_input, prompt_type, prompts, sys_prompt2=None):
    input_text = sys_prompt + "\n" + prompts.get(prompt_type) + "\n" + user_input
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


# evalutes the performance of the baseline model on the given datasets with different prompting
    # args: 
    #     datasets (dict): dictionary of datasets for each category
    # results: none (saves to json file in results dir)
def evaluate_prompting(datasets, results_dir="results/prompting"):
    results = {}
    prompt_types = ["zs", "os", "fs"] # zero shot, one shot, few shot
    prompt_files = ["prompts/prompt_zs.txt", "prompts/prompt_os.txt", "prompts/prompt_fs.txt"]
    prompts = load_prompts(prompt_files)
    
    for prompt_type in tqdm(prompt_types, total=len(prompt_types), desc="Prompt Types"): # try all combinations of prompt types
        results[prompt_type] = {}

        for category, dfs in datasets.items():
            if category == "pi_class":
                handle_classification_task_prompting(dfs, category, results[prompt_type], prompt_type, prompts)
                        
            elif category in ["pi_detect", "jailbreak"]:
                handle_detection_task_prompting(dfs, category, results[prompt_type], prompt_type, prompts)

            elif category == "password":
                handle_password_task_prompting(dfs, category, results[prompt_type], prompt_type, prompts)
            
        # save results as a json
        with open(os.path.join(results_dir, f"{prompt_type}_metrics.json"), "w") as f:
            json.dump(results[prompt_type], f, indent=4)
        print(f"Prompting results saved at results/prompting/{prompt_type}_metrics.json")


# helper function for evaluate_prompting that works on the pi_class datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    # returns: none - updates results without returning
def handle_classification_task_prompting(dfs, category, results, prompt_type, prompts):
    y_true = []
    y_pred = []
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            output = get_prompting_output(row["sys_prompt"], row["user_input"], prompt_type, prompts)
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

    print("PROMPTING", category)
    print("    ACCURACY: ", accuracy)
    print("    PREC, RECALL, F1: ", stats)
    print("    CONF MATRIX: ", cm)


# helper function for evaluate_baseline that works on the jailbreak and pi_detect datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    # returns: none - updates results without returning
def handle_detection_task_prompting(dfs, category, results, prompt_type, prompts):
    correct = 0
    total = 0
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            output = get_prompting_output(row["sys_prompt"], row["user_input"], prompt_type, prompts)
            if evaluate_output(output):
                correct += 1
            total += 1
    accuracy = correct / total
    results[category] = {"accuracy": accuracy}

    print("PROMPTING", category)
    print("    ACCURACY: ", accuracy)


# helper function for evaluate_prompting that works on the password datasets
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results
    # returns: none - updates results without returning
def handle_password_task_prompting(dfs, category, results, prompt_type, prompts):
    correct = 0
    total = 0
    for df in dfs:
        for _, row in tqdm(df.iterrows(), desc=f"Processing {category} Rows", total=len(df), leave=False):
            sys_prompt1 = row["sys_prompt1"] if "sys_prompt1" in row else row["sys_prompt"]
            sys_prompt2 = row["sys_prompt2"] if "sys_prompt2" in row else None
            user_input = row["user_input"]

            output = get_prompting_output(sys_prompt1, user_input, prompt_type, prompts, sys_prompt2=sys_prompt2)
            if evaluate_output(output, row["password"]):
                correct += 1
            total += 1
    accuracy = correct / total
    results[category] = {"accuracy": accuracy}

    print("PROMPTING", category)
    print("    ACCURACY: ", accuracy)


############################## MAIN METHOD ##############################

def main():
    # train = load_split("train")
    # validation = load_split("validation")
    validation_spotlighting = load_split("validation_spotlighting")
    # test = load_split("test")
    print("Datasets loaded")

    # evaluate_prompting(validation_spotlighting)
    evaluate_spotlighting(validation_spotlighting)

if __name__ == "__main__":
    main()