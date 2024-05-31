# CatchMeIfYouDAN

Alice Guo, Grace Jin, Jenny Wei

Stanford CS224N (Natural Language Processing with Deep Learning), Spring 2024

## DATASETS + SETUP

### Raw Datasets
This project uses a total of 8 datasets from various sources. 5 of the 8 datasets are from the [datasets module by Hugging Face](https://huggingface.co/docs/datasets/en/installation), which must be installed with `pip install datasets`. These are:

1. Deepset prompt injections [dataset](https://huggingface.co/datasets/deepset/prompt-injections)
2. HackAPrompt project injections [dataset](https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset)
3. Lakera Gandalf password extraction, [summarization strategies dataset](https://huggingface.co/datasets/Lakera/gandalf_summarization?row=95)
4. Lakera Gandalf password extraction, [ignore instructions strategies dataset](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions)
5. Lakera Mosscap password extraction [dataset](https://huggingface.co/datasets/Lakera/mosscap_prompt_injection)

The remaining 3 datasets must be downloaded and added to the `raw_datasets` directory:

6. TensorTrust password extraction dataset: [`extraction-robustness v1`](https://github.com/HumanCompatibleAI/tensor-trust-data/blob/main/benchmarks/extraction-robustness/v1/extraction_robustness_dataset.jsonl)
7. Do Anything Now jailbreak dataset: [`forbidden_question_set.csv.zip`](https://github.com/verazuo/jailbreak_llms/blob/main/data/forbidden_question_set.csv.zip) - make sure to unzip before adding
8. ProtectAI jailbreak dataset: [`jailbreak.json`](https://github.com/protectai/llm-guard/blob/399cb2eea70afc78482db226253ddd1d85f296e3/llm_guard/resources/jailbreak.json)

### PreProcessing
Then, run `python process_datasets.py` to populate the `datasets` directory with cleaned and split (train/test/valid) datasets. Since `process_datasets.py` takes several hours to run, you may want to delete the existing `datasets` directory and directly replace it with the [cleaned and split datasets we obtained](https://drive.google.com/file/d/1_7pvC6xR-JrQ0l1QbeFhOroAizhbA4Vw/view?usp=sharing) instead.

### Dataset Combination
Although this project uses 8 datasets, we will combine performance metrics across several datasets according to the following categories:

* **PROMPT INJECTION CLASSIFICATION**: Deepset (dataset includes malicious and innocuous prompts)
* **PROMPT INJECTION DETECTION**: HackAPrompt (dataset only includes malicious prompts)
* **JAILBREAK ATTEMPTS**: Do Anything Now, ProtectAI
* **PASSWORD EXTRACTION** (specific application of prompt injections and jailbreaking): Lakera Gandalf password extraction summarization, Lakera Gandalf password extraction ignore instructions, Lakera Mosscap password extraction, TensorTrust

## EXPERIMENTS + RESULTS
