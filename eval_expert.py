import json
import pandas as pd

def load_predictions(filepath):
    predictions = pd.read_csv(filepath)
    predictions['Similarity Score'] = predictions['Similarity Score'].str.strip('[]').astype(float)
    predictions = predictions['Similarity Score'].tolist()
    return predictions

def load_true_labels(filepath):
    true_labels = pd.read_parquet(filepath)
    true_labels = true_labels['label'].tolist()
    return true_labels, len(true_labels) # length for weighting

def calculate_accuracy(predictions, true_labels):
    # threshold for binary classification
    binary_predictions = [1 if score > 0.5 else 0 for score in predictions]
    correct_predictions = sum(1 for true, pred in zip(true_labels, binary_predictions) if true == pred)
    return correct_predictions / len(true_labels)

def evaluate_accuracy_for_category(test_valid, dataset_cats):
    accuracies = {}
    for category, datasets in dataset_cats.items():
        category_accuracies = []
        total_samples = 0
        weighted_sum = 0
        for dataset in datasets:
            predictions_file = f'dansn_model/{test_valid}_preds/{dataset}_dansn.csv' # for danSN save format

            true_labels_file = f'datasets/{dataset}/{test_valid}.parquet'
            predictions = load_predictions(predictions_file)
            true_labels, num_samples = load_true_labels(true_labels_file)
            accuracy = calculate_accuracy(predictions, true_labels)
            category_accuracies.append((dataset, accuracy, num_samples))
            weighted_sum += accuracy * num_samples
            total_samples += num_samples

        weighted_accuracy = weighted_sum / total_samples if total_samples > 0 else 0
        accuracies[category] = (category_accuracies, weighted_accuracy)
    return accuracies

############################## RUN ACC EVALS ##############################

dataset_cats = {
    "pi_class": ["pi_deepset"],
    "pi_detect": ["pi_hackaprompt"],
    "jailbreak": ["dan_jailbreak", "protectai_jailbreak"],
    "password": ["lakera_ignore", "lakera_summ", "lakera_mosscap", "tensortrust_extraction"]
}


category_accuracies = evaluate_accuracy_for_category("validation", dataset_cats)
for category, (results, weighted_acc) in category_accuracies.items():
    print(f"Category: {category}, Weighted Accuracy: {weighted_acc:.4f}")
    for dataset, accuracy, num_samples in results:
        print(f"  Dataset: {dataset}, Accuracy: {accuracy:.4f}, Samples: {num_samples}")