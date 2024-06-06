import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
import pandas as pd

############################## DEFINING CONSTANTS ##############################
dataset_cats = {
    "pi_class": ["pi_deepset"],
    "pi_detect": ["pi_hackaprompt"],
    "jailbreak": ["dan_jailbreak", "protectai_jailbreak"],
    "password": ["lakera_ignore", "lakera_summ", "lakera_mosscap", "tensortrust_extraction"]
}

results_dir="results/danbert"
dataset_dir = "datasets"
danbert_path = "danbert_model/danbert.bin"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
danbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
danbert_model = torch.load(danbert_path, map_location=device)
danbert_model.to(device)

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

# gets model predictions for a set of user inputs
    # arg: df series of user_inputs
    # returns: model predictions and probabilities of predictions
def get_danbert_preds(user_inputs, batch_size=32):
    probabilities = []
    predictions = []
    
    for i in tqdm(range(0, len(user_inputs), batch_size)):
        batch = user_inputs[i:i + batch_size]
        inputs = danbert_tokenizer(batch.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = danbert_model(**inputs)
        logits = outputs.logits
        batch_probabilities = torch.sigmoid(logits)
        batch_predictions = torch.argmax(logits, dim=-1)
        probabilities.extend(batch_probabilities.cpu().tolist())
        predictions.extend(batch_predictions.cpu().tolist())
    
    return probabilities, predictions

# evalutes the performance of danbert on the given datasets
    # args: 
    #     datasets (dict): dictionary of datasets for each category
    # results: none (saves to json file in results dir)
def evaluate_danbert(datasets, split):
    results = {}

    for category, dfs in datasets.items():
        # all pi_class datasets are classification tasks, so we analyze all true and mis classifications
        # store precision, accuracy, recall, f1, and confusion matrix metrics
        eval_category(dfs, category, results, split)

    # save results as a json
    with open(os.path.join(results_dir, f"danbert_metrics_{split}.json"), "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved at results/danbert_metrics.json")


# helper function for evaluate_danbert that evaluates a single category
    # args: 
    #     dfs (list of DataFrame) - list of dfs
    #     category (str) - the category of the dataset being processed
    #     results (dict) - dictionary to store results

    # returns: none
def eval_category(dfs, category, results, split):
    y_true = []
    y_pred = []

    dataset_names = dataset_cats[category]
    for i in tqdm(range(len(dfs))):
        df = dfs[i]
        df_name = dataset_names[i]
        probabilities, outputs = get_danbert_preds(df["user_input"])
        y_pred.extend(outputs)
        y_true.extend(df["label"].tolist())
        output_df = pd.DataFrame({
        'Pred_Labels': outputs,
        'Confidence': probabilities
        })
        path = os.path.join(results_dir, f"{df_name}_preds_{split}.csv")
        output_df.to_csv(path)

    stats = classification_report(y_true, y_pred, output_dict=True)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    results[category] = {
        "stats": stats,
        "accuracy": accuracy,
        "cm": cm.tolist()
    }
    print("    ACCURACY: ", accuracy)
    print("    PREC, RECALL, F1: ", stats)
    print("    CONF MATRIX: ", cm)

######################## RUN EVAL ###############
if __name__ == "__main__":
    val_dfs = load_split("validation")
    test_dfs = load_split("test")
    evaluate_danbert(val_dfs, "validation")
    evaluate_danbert(test_dfs, "test")