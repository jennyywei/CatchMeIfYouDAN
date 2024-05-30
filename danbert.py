import pandas as pd
import torch
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from itertools import combinations
from pprint import pprint
import os
import json

############################## GLOBAL VARIABLES AND DEFS ##############################

dataset_name = "pi_deepset"

dataset_dir = "datasets"

results_dir = "results"

MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 150
LEARNING_RATE = 1e-05

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

############################## LOADING FILES AND TOOLS ##############################

def load_split(split):
    """
    loads pi_deepset split (train, valid, or test)
    arg: split (str) - dataset split to load (train, validation, or test)
    returns: dataset in df format
    """
    if (split != "train" and split != "validation" and split != "test"):
        print("Tried to load an invalid split")
        return
    
    path = os.path.join(dataset_dir, dataset_name)
    file_path = os.path.join(path, f"{split}.parquet")
    if os.path.exists(file_path):
        return pd.read_parquet(file_path, columns=["user_input", "label"])
    else:
        print(f"{dataset_name} {split} split not found when loading dataset")

############################## DEEPSET DATASET CLASS ##############################

class Deepset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        user_input = str(self.data.user_input[index])
        user_input = " ".join(user_input.split()) # standardize white-space
        inputs = self.tokenizer.encode_plus(
            user_input,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len

############################## RUN FINE-TUNING ##############################

train = load_split("train")
validation = load_split("validation")

training_set = Deepset(train, tokenizer, MAX_LEN)
testing_set = Deepset(validation, tokenizer, MAX_LEN)

train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0, # TODO: change if running on GPU?
}

test_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0, # TODO: change if running on GPU?
}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def calculate_accuracy(big_idx, targets):
    n_correct = (big_idx==targets).sum().item()
    return n_correct

def train(epoch):
    tr_loss = 0
    n_correct = 0
    n_steps = 0
    n_examples = 0
    model.train()
    for i, data in enumerate(tqdm(training_loader)):
        ids = data['ids']
        mask = data['mask']
        targets = data['targets']

        outputs = model(ids, mask).logits
        loss = loss_function(outputs, targets)
        tr_loss += loss.item()
        max_val, max_idx = torch.max(outputs.data, dim=1)
        n_correct += calculate_accuracy(max_idx, targets)

        n_steps += 1
        n_examples += targets.size(0)
        
        if i % 5000 == 0:
            loss_step = tr_loss / n_steps
            accu_step = (n_correct*100) / n_examples 
            print(f"Step {i}, Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")

        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        # optimizer.step() 

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100) / n_examples}')
    epoch_loss = tr_loss / n_steps
    epoch_accu = (n_correct * 100) / n_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    return 

for epoch in tqdm(range(EPOCHS)):
    train(epoch)

def valid(model, testing_loader):
    model.eval()
    val_loss = 0
    n_correct = 0
    n_steps = 0
    n_examples = 0
    with torch.no_grad():
        for _, data in enumerate(tqdm(testing_loader)):
            ids = data['ids']
            mask = data['mask']
            targets = data['targets']
            outputs = model(ids, mask).logits
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            max_val, max_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(max_idx, targets)

            n_steps += 1
            n_examples += targets.size(0)
            
            if _%5000==0:
                loss_step = val_loss / n_steps
                accu_step = (n_correct*100) / n_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = val_loss / n_steps
    epoch_accu = (n_correct*100) / n_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_accu

accuracy = valid(model, testing_loader)
print("Accuracy on test data = %0.2f%%" % accuracy)

output_model_path = 'models/pytorch_distilbert.bin'
output_vocab_path = 'models/vocab_distilbert.bin'

model_to_save = model
torch.save(model_to_save, output_model_path)
tokenizer.save_vocabulary(output_vocab_path)
