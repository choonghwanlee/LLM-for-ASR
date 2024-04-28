import torch
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BertTokenizer, BertForMaskedLM
import werpy
import jiwer
from evaluate import load
import re

wer_standardize = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.ExpandCommonEnglishContractions(),
        jiwer.RemoveKaldiNonWords(),
        jiwer.RemoveSpecificWords(["uh", "um", "mm"]), 
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

def normalize(input):
    input = werpy.normalize(input)
    input = wer_standardize(input)
    input = ' '.join([' '.join(sublist) for sublist in input])
    return input

# Step 1: Load the dataset
dataset = load_dataset('edinburghcstr/edacc')
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

# Filter the dataset to include only samples with ground truth text longer than 10 words
def filter_function(sample):
    ground_truth = sample['text']
    return 15 <= len(ground_truth.split()) <= 100

filtered_dataset = dataset.filter(filter_function)

# Step 2: Load the Whisper model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model_whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

## change configs to allow logits and scores
model_whisper.generation_config.output_logits = True

# Load BERT tokenizer and model
tokenizer_bert = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
model_bert = BertForMaskedLM.from_pretrained("google-bert/bert-base-uncased").to(device)

# Define uncertainty threshold and top N predictions
uncertainty_threshold = 0.5
top_n_predictions = 5

def contains_number(text):
    return bool(re.search(r'\d', text))

def extract_uncertain(audio_sample):
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]

    # Preprocess audio input
    inputs = processor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    )
    
    # Ensure the input tensor has the expected shape
    input_features = inputs.input_features.to(device)

    # Generate transcription
    with torch.no_grad():
        output = model_whisper.generate(
            input_features=input_features,
            generation_config=model_whisper.generation_config,
            task='transcribe',
            language='english',
            return_dict_in_generate=True
        )
        logits = output['logits']
        transcription = processor.batch_decode(output['sequences'], skip_special_tokens=True)[0]

        # Extract uncertain tokens
        uncertain_indices = []
        for i, logit in enumerate(logits):
            if torch.max(logit).item() < uncertainty_threshold:
                uncertain_indices.append(i)
    
    return transcription, uncertain_indices

def predict_masks(masked_text, uncertain_indices, top_n=top_n_predictions):
    input_ids = tokenizer_bert(masked_text, return_tensors="pt").input_ids.to(device)
    logits = model_bert(input_ids).logits

    predicted_tokens = []
    for idx, uncertain_idx in enumerate(uncertain_indices):
        mask_logits = logits[0, uncertain_idx]
        top_n_logits, top_n_ids = torch.topk(mask_logits, top_n, dim=-1)
        predicted_tokens.append([tokenizer_bert.convert_ids_to_tokens([id])[0] for id in top_n_ids.tolist()])
        
    return predicted_tokens

def replace_uncertain_tokens(original_text, uncertain_indices, predicted_tokens):
    words = original_text.split()
    for i, idx in enumerate(uncertain_indices):
        words[idx] = predicted_tokens[i][0]  # Replace with the token with highest logit value
    return " ".join(words)

wer_scores = []
wil_scores = []
mer_scores = []
num_words = []

for sample in filtered_dataset["test"]:
    audio_sample = sample['audio']  # audio feature
    ground_truth = sample['text']  # ground truth transcription
    accent = sample['accent']  # speaker accent 
    
    # Extract uncertain tokens
    transcription, uncertain_indices = extract_uncertain(audio_sample)
    
    # Predict top-N tokens using BERT MLM
    predicted_tokens = predict_masks(transcription, uncertain_indices)
    
    # Replace uncertain tokens with predicted tokens
    corrected_text = replace_uncertain_tokens(transcription, uncertain_indices, predicted_tokens)
    normalized_ground_truth = normalize(ground_truth)
    normalized_corrected_text = normalize(corrected_text)
    
    # Calculate number of words
    num_words.append(len(normalized_ground_truth.split()))
    
    # Calculate WER
    wer_score = jiwer.wer(normalized_ground_truth, normalized_corrected_text)
    wer_scores.append(wer_score)
    
    # Calculate WIL
    wil_score = jiwer.wil(normalized_ground_truth, normalized_corrected_text)
    wil_scores.append(wil_score)
    
    # Calculate MER
    mer_score = jiwer.mer(normalized_ground_truth, normalized_corrected_text)
    mer_scores.append(mer_score)

average_wer = sum(wer_scores) / len(wer_scores)
average_wil = sum(wil_scores) / len(wil_scores)
average_mer = sum(mer_scores) / len(mer_scores)

print("Average WER:", average_wer)
print("Average WIL:", average_wil)
print("Average MER:", average_mer)

