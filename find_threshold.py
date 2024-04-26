from transformers import WhisperProcessor, WhisperForConditionalGeneration
import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt

# Load the Whisper model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Load the Hugging Face dataset
dataset = datasets.load_dataset("Nexdata/accented_english", split="train")  

# Define a function to preprocess the data
def preprocess(batch):
    audio = batch["audio"]["array"]
    input_features = processor(audio, return_tensors="pt")
    return input_features

# Preprocess the dataset
processed_dataset = dataset.map(preprocess, batched=True, batch_size=8)

# List to store the maximum predicted probabilities
max_probs = []

# Iterate through the preprocessed dataset
for batch in processed_dataset:
    input_features = batch
    
    # Forward pass through the model
    with torch.no_grad():
        logits = model(**input_features).logits
    
    # Calculate probabilities and find the maximum probability for each token
    probs = torch.softmax(logits, dim=-1)
    max_probs.extend(probs.max(-1)[0].tolist())

# Plot the distribution of maximum predicted probabilities
plt.hist(max_probs, bins=20)
plt.title("Distribution of Maximum Predicted Probabilities")
plt.xlabel("Maximum Probability")
plt.ylabel("Frequency")
plt.show()

