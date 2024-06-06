import pandas as pd
import torch
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, pipeline
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
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
EPOCHS = 10
BATCH_SIZES = [4, 8, 16]
LEARNING_RATES= [5e-5, 4e-5, 3e-5, 2e-5, 1e-5]
PATIENCE = 3

############################### DEVICE AND MODEL SET UP #######################################
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased").to(device)

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
            padding=False,
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

############################## TRAINING FUNCTION DEFINITIONS ##############################

def collate_fn(batch):
    ids = [item['ids'] for item in batch]
    mask = [item['mask'] for item in batch]
    targets = [item['targets'] for item in batch]

    ids_padded = pad_sequence(ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    mask_padded = pad_sequence(mask, batch_first=True, padding_value=0)
    targets = torch.stack(targets)

    return {
        'ids': ids_padded,
        'mask': mask_padded,
        'targets': targets
    }

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
        ids = data['ids'].to(device, dtype = torch.long)
        mask = data['mask'].to(device, dtype = torch.long)
        targets = data['targets'].to(device, dtype = torch.long)

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
        # optimizer.step() # TODO: uncomment when running on GPU

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100) / n_examples}')
    epoch_loss = tr_loss / n_steps
    epoch_accu = (n_correct * 100) / n_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")
    return epoch_loss, epoch_accu


def valid(model, testing_loader):
    model.eval()
    val_loss = 0
    n_correct = 0
    n_steps = 0
    n_examples = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(testing_loader)):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).logits
            loss = loss_function(outputs, targets)
            val_loss += loss.item()
            max_val, max_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_accuracy(max_idx, targets)

            n_steps += 1
            n_examples += targets.size(0)
            
            if i % 5000==0:
                loss_step = val_loss / n_steps
                accu_step = (n_correct*100) / n_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = val_loss / n_steps
    epoch_accu = (n_correct*100) / n_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")
    
    return epoch_loss, epoch_accu

############################## LOAD DATASET ##############################
results = {
    f"batch_size_{bs}_lr_{lr}": {
        'train_losses': {},
        'train_accuracies': {},
        'val_losses': {},
        'val_accuracies': {}
    }
    for bs in BATCH_SIZES for lr in LEARNING_RATES
}

train_df = load_split("train")
validation_df = load_split("validation")

############################## RUN FINE-TUNING ##############################
for bs in BATCH_SIZES:
    for lr in LEARNING_RATES:
        print(f"\nTesting with batch size: {bs}, learning rate: {lr}\n")
        key = f"batch_size_{bs}_lr_{lr}"
        training_set = Deepset(train_df, tokenizer, MAX_LEN)
        testing_set = Deepset(validation_df, tokenizer, MAX_LEN)

        train_params = {
            'batch_size': bs,
            'shuffle': True,
            'num_workers': 0, # TODO: change if running on GPU?
            'collate_fn': collate_fn
        }

        test_params = {
            'batch_size': bs,
            'shuffle': True,
            'num_workers': 0, # TODO: change if running on GPU?
            'collate_fn': collate_fn
        }

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params =  model.parameters(), lr=lr)

        train_losses = {}
        train_accuracies = {}
        val_losses = {}
        val_accuracies = {}

        best_val_loss = float('inf')  
        patience_counter = 0
        # training          
        for epoch in tqdm(1, range(EPOCHS) + 1):
            train_loss, train_accu = train(epoch)
            val_loss, val_accu = valid(model, testing_loader)

            # store metrics
            train_losses[epoch] = train_loss
            train_accuracies[epoch] = train_accu
            val_losses[epoch] = val_loss
            val_accuracies[epoch] = val_accu
            
            # early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'models/best_model_{key}.pth')
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        results[key]['train_losses'] = train_losses
        results[key]['train_accuracies'] = train_accuracies
        results[key]['val_losses'] = val_losses
        results[key]['val_accuracies'] = val_accuracies

        model.load_state_dict(torch.load(f'models/best_model_{key}.pth'))
        loss, accuracy = valid(model, testing_loader)
        print("Accuracy on test data = %0.2f%%" % accuracy)

        output_model_path = f'models/pytorch_distilbert_{key}.bin'
        output_vocab_path = f'models/vocab_distilbert_{key}.bin'

        model_to_save = model
        torch.save(model_to_save, output_model_path)
        tokenizer.save_vocabulary(output_vocab_path)

with open('models/training_metrics.json', 'w') as f:
    json.dump(results, f)