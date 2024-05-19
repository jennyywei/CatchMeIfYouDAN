import os
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

llm = pipeline('text-generation', model='gpt2')

def llm_input(sys_prompt, user_input):
    return f"{sys_prompt} {user_input}"

def evaluate_response(llm_output, label):
    return llm_output == label

def process_dataset(name, file_path, is_classification=False):
    df = pd.read_parquet(file_path)
    results = []

    for index, row in df.iterrows():
        if (name == "tensortrust_extraction"):
            sys_prompt1 = row['sys_prompt1']
            user_input = row['user_input']
            sys_prompt2 = row['sys_prompt2']
            password = row['label']
        else:
            sysys_prompt = row['sys_prompt']
            user_input = row['user_input']
            label = row['label']

        # Process the inputs through the LLM
        processed_input = llm_input(sys_prompt, user_input)
        if is_classification:
            # Use classification pipeline for binary classification problem
            llm_output = classification_pipeline(processed_input)[0]['label']
            llm_output = int(llm_output.split('_')[-1])
        else:
            # Use text generation pipeline for non-classification problems
            llm_output = llm(processed_input, max_length=50, num_return_sequences=1)[0]['generated_text']

        # Evaluate the response
        if is_classification:
            results.append((llm_output, label))
        else:
            result = evaluate_response(llm_output, label)
            results.append(result)

    return results

def plot_classification_metrics(y_true, y_pred, dataset_name, split):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Results for {dataset_name} - {split}:")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{cm}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {dataset_name} - {split}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def main():
    datasets_folder = 'datasets'
    dataset_names = ['dan_jailbreak', 'lakera_ignore', 'lakera_mosscap', 'lakera_summ', 'pi_deepset', 'pi_hackaprompt', 'protectai_jailbreak', 'tensortrust_extraction']

    for dataset_name in dataset_names:
        dataset_path = os.path.join(datasets_folder, dataset_name)
        
        for split in ['train.parquet', 'test.parquet', 'validation.parquet']:
            file_path = os.path.join(dataset_path, split)
            is_classification = dataset_name == 'pi_deepset'
            results = process_dataset(file_path, is_classification)

            if is_classification:
                y_pred, y_true = zip(*results)
                plot_classification_metrics(y_true, y_pred, dataset_name, split)
            else:
                accuracy = sum(results) / len(results)
                print(f"Accuracy for {dataset_name} - {split}: {accuracy}")

if __name__ == "__main__":
    main()