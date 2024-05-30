import torch
from transformers import AutoTokenizer, AutoModel

# Load pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def get_embeddings(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs.get('token_type_ids', None)
    
    if token_type_ids is not None:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    return outputs.last_hidden_state.mean(dim=1)  # mean pooling

# Function to compute dot product
def compute_dot_product(embedding1, embedding2):
    return torch.dot(embedding1.squeeze(), embedding2.squeeze())

# Function to predict relationship
def predict_relationship(dot_product, threshold=0.5):
    return dot_product.item() > threshold

# Function to process and test model on inputs
def test_model(system_prompt, vector_prompt, test_inputs):
    # Compute embeddings for system and vector prompts
    system_embedding = get_embeddings(system_prompt)
    vector_embedding = get_embeddings(vector_prompt)
    
    # Compute dot product for system and vector prompts
    dot_product = compute_dot_product(system_embedding, vector_embedding)
    is_malicious = predict_relationship(dot_product)
    
    # Test the model on test inputs
    results = []
    for test_input in test_inputs:
        test_embedding = get_embeddings(test_input)
        test_dot_product = compute_dot_product(system_embedding, test_embedding)
        results.append(predict_relationship(test_dot_product))
    
    return is_malicious, results

# Example inputs
system_prompt = "Your system prompt text here"
vector_prompt = "Your vector prompt text here"
test_inputs = ["Test input 1", "Test input 2", "Test input 3"]

# Run the model
is_malicious, test_results = test_model(system_prompt, vector_prompt, test_inputs)

# Print results
print(f"Is malicious: {is_malicious}")
print(f"Test results: {test_results}")