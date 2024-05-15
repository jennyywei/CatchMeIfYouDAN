# CatchMeIfYouDAN

Alice Guo, Grace Jin, Jenny Wei

Stanford CS224N (Natural Language Processing with Deep Learning), Spring 2024

## DATASETS
This project uses a total of 8 datasets from various sources. 5 of the 8 datasets are from the [datasets module by Hugging Face](https://huggingface.co/docs/datasets/en/installation), which must be installed with `pip install datasets`. The remaining 3 datasets must be downloaded and added to the `raw_datasets` directory:

* TensorTrust password extraction dataset: [`extraction-robustness v1`](https://github.com/HumanCompatibleAI/tensor-trust-data/blob/main/benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl)
* ProtectAI jailbreak dataset: [`jailbreak.json`](https://github.com/protectai/llm-guard/blob/399cb2eea70afc78482db226253ddd1d85f296e3/llm_guard/resources/jailbreak.json)
* Do Anything Now jailbreak dataset: [`forbidden_question_set.csv.zip`](https://github.com/verazuo/jailbreak_llms/blob/main/data/forbidden_question_set.csv.zip) - make sure to unzip before adding

Then, run `python process_datasets.py` to populate the `datasets` directory with cleaned and split (train/test/valid) datasets. Since `process_datasets.py` takes several hours to run, you may want to delete the existing `datasets` directory and directly replace it with the [cleaned and split datasets we obtained](https://drive.google.com/file/d/1zIETkJ7Y1iIQ9bKcIgceaFSlX0vX4gPr/view?usp=sharing) instead.

Although this project uses 8 datasets, we will combine performance metrics across several datasets according to the following framework:

[insert framework later]
