import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

datasets = ["pi_class", "pi_detect", "jailbreak", "password"]

results_dir = "results/prompting"
bargraph_order = ["zs", "os", "fs"]

# results_dir = "results/spotlighting"
# bargraph_order = ["dl", "dm", "ec", "dl_dm", "dl_ec", "dm_ec", "dl_dm_ec"]

def load_results(dir):
    data = {}
    for filename in os.listdir(dir):
        if filename.endswith("_metrics.json"):
            model_name = "_".join(filename.split("_")[:-1])
            with open(os.path.join(dir, filename), "r") as f:
                data[model_name] = json.load(f)
    return data

def accuracy(results):
    model_names = list(results.keys())
    accuracies = {dataset: [] for dataset in datasets}

    for dataset in datasets:
        for model in model_names:
            accuracies[dataset].append(results[model][dataset]["accuracy"])

    # plot bar graphs for each dataset
    for dataset in datasets:
        plt.figure(figsize=(10, 6), dpi=250)
        ax = sns.barplot(x=model_names, y=accuracies[dataset], order=bargraph_order)
        for p in ax.patches: # annotate bars
            ax.annotate(f"{p.get_height():.4f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha="center", va="baseline", fontsize=12, color="black", xytext=(0, 5), 
                        textcoords="offset points")
        
        plt.title(f"Accuracy Comparison for {dataset}")
        plt.xlabel("Models")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.savefig(os.path.join(results_dir, f"acc_{dataset}.png"))

def confusion_mat(results):
    model_names = list(results.keys())
    for model in model_names:
        cm = np.array(results[model]["pi_class"]["cm"])        
        precision = results[model]["pi_class"]["stats"]["weighted avg"]["precision"]
        recall = results[model]["pi_class"]["stats"]["weighted avg"]["recall"]
        f1_score = results[model]["pi_class"]["stats"]["weighted avg"]["f1-score"]
        
        plt.figure(figsize=(10, 6), dpi=250)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f"Confusion Matrix for {model} on pi_class Dataset")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        
        # adding weighted precision, recall, and f1-score to the plot
        plt.gcf().text(0.7, 0.8, f"Weighted Avg Precision: {precision:.2f}", fontsize=12, verticalalignment="center")
        plt.gcf().text(0.7, 0.7, f"Weighted Avg Recall: {recall:.2f}", fontsize=12, verticalalignment="center")
        plt.gcf().text(0.7, 0.6, f"Weighted Avg F1-score: {f1_score:.2f}", fontsize=12, verticalalignment="center")
        
        plt.tight_layout(rect=[0, 0, 0.65, 1])
        plt.savefig(os.path.join(results_dir, f"cm_{model}.png"))

def main():
    results = load_results(results_dir)
    accuracy(results)
    confusion_mat(results)

if __name__ == "__main__":
    main()