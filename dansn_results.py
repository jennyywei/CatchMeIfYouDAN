import matplotlib.pyplot as plt
import pandas as pd

############################## GLOBAL VARIABLES AND DEFS ##############################

epoch_losses_path = "dansn_training/epoch_vals.csv"
batch_losses_path = "dansn_training/batch_vals.csv"
n = 1000

############################## PLOT TRAINING/VALIDATION LOSSES PER EPOCH ##############################

data = pd.read_csv(epoch_losses_path, skiprows=1)  # skip header
data.columns = ["Epoch", "Training Loss", "Validation Loss"]

# plot
plt.figure(figsize=(10, 5), dpi=250)
plt.plot(data["Epoch"], data["Training Loss"], label="Training Loss")
plt.plot(data["Epoch"], data["Validation Loss"], label="Validation Loss")
plt.title("Training vs. Validation Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/dansn/epoch_loss.png") 

############################## PLOT TRAINING LOSSES PER BATCH ##############################

data = pd.read_csv(batch_losses_path)  # skip header
data.columns = ["Epoch", "Batch", "Training Loss", "Training Accuracy"]

# plot
plt.figure(figsize=(10, 6), dpi=250)
plt.plot(data["Batch"][:n], data["Training Loss"][:n], linestyle='-', color='blue')
plt.title('Training Loss Over Batches')
plt.xlabel('Batch Number')
plt.ylabel('Training Loss')
plt.grid(True)
plt.savefig("results/dansn/training_loss.png") 