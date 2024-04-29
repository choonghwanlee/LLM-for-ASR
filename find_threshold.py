from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

# Step 1: Load the dataset
dataset = load_dataset('edinburghcstr/edacc') ## install from HF instead
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) ## resample to 16 kHz

# Shuffle the validation set
validation_set = dataset['validation']
validation_set = validation_set.shuffle(seed=42)  # Set a seed for reproducibility

# Select 1000 random entries from the shuffled validation set
subset = validation_set.select(range(1000))

## sample usage
sample = dataset["validation"][0] ## sample a row from the validation dataset
audio_sample = sample['audio'] ## audio feature
ground_truth = sample['text'] ## ground truth transcription
accent = sample['accent'] ## speaker accent

# Step 2: Load the Whisper model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

## change configs to allow logits and scores
model.generation_config.output_logits = True

# Step 3: Preprocess the resampled audio data
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]

audio_input = processor(
    waveform,
    sampling_rate = sampling_rate,
    return_tensors="pt"
).input_features

audio_input = audio_input.to(device)

# Step 4: Generate the transcription
with torch.no_grad():
    ## output is a dictionary with keys ['sequences', 'logits', 'past_key_values']. we only care about the first two
    output = model.generate(input_features=audio_input, generation_config= model.generation_config, task='transcribe', language='english', return_dict_in_generate=True) ## generate results

## get the actual prediction
transcription = processor.batch_decode(output['sequences'], skip_special_tokens=True)[0]
print('Sample Transcription Results: ', transcription)
print('Ground Truth Results: ', ground_truth)

# normalized_probs = [F.softmax(logit) for logit in output['logits']] ## obtain normalized probabilities for each token in transcription
# max_prob_per_token = [torch.max(probs).item() for probs in normalized_probs] ### model's confidence on a token

###############################

def _process_logits(logits):
    normalized_probs = [F.softmax(logit) for logit in logits] ## obtain normalized probabilities for each token in transcription
    max_prob_per_token = [torch.max(probs).item() for probs in normalized_probs] ### model's confidence on a token
    ## a list of highest probabilities for each token in a sample
    return max_prob_per_token

distribution = []
# num_correct = []
def _get_max_probs(examples):
    waveforms = [x["array"] for x in examples["audio"]]
    sampling_rate = examples['audio'][0]['sampling_rate']
    input_features = processor.feature_extractor(waveforms, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
    outputs = model.generate(input_features, generation_config= model.generation_config, task='transcribe', language='english', return_dict_in_generate=True, use_cache=True)
    batch_probs = [_process_logits(logits) for logits in outputs['logits']] ## list of lists of highest probabilities for each token in a sample 
    distribution.extend([probs for batch in batch_probs for probs in batch]) ## flatten and add to global list


subset.map(lambda example: _get_max_probs(example), batched=True, batch_size=4)

print(len(distribution))

# Plot the distribution of maximum predicted probabilities
plt.figure(figsize=(10, 6))
plt.hist(distribution, bins=100, color='skyblue', alpha=0.7)
plt.xlabel('Maximum Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Maximum Predicted Probabilities')
plt.grid(True)
plt.show()