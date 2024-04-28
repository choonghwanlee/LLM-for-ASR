## Approach 3: Exclusively using BERT MLM predictions to evaluate the uncertain token

import whisper
import torch
from transformers import BertTokenizer, BertForMaskedLM
from datasets import load_dataset
from jiwer import wer, wil, mer

# Load Whisper model
model_whisper = whisper.load_model("base")


# Function to transcribe audio with Whisper
def transcribe_audio(audio_path):
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model_whisper.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model_whisper, mel, options)
    return result.text, result.tokens


# Load BERT model and tokenizer for Masked LM
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model_bert = BertForMaskedLM.from_pretrained("bert-base-uncased")


# Function to mask uncertain tokens
def mask_uncertain_tokens(transcription, tokens, threshold=0.5):
    masked_transcription = []
    words = transcription.split()
    for word, token in zip(words, tokens):
        if token < threshold:
            masked_transcription.append("[MASK]")
        else:
            masked_transcription.append(word)
    return " ".join(masked_transcription)


# Function to predict masked tokens using BERT
def predict_masks(masked_text):
    input_ids = tokenizer(masked_text, return_tensors="pt").input_ids
    logits = model_bert(input_ids).logits
    mask_token_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    # Get the predicted token IDs
    predicted_token_ids = logits.argmax(dim=2)[
        0, mask_token_index
    ].tolist()  # Simplified and corrected
    predicted_tokens = [tokenizer.decode([id]).strip() for id in predicted_token_ids]

    # Replace [MASK] with predicted tokens
    words = masked_text.split()
    mask_counter = 0  # To keep track of which [MASK] to replace
    for i, word in enumerate(words):
        if word == "[MASK]":
            words[i] = predicted_tokens[mask_counter]
            mask_counter += 1

    return " ".join(words)


# Evaluate transcription accuracy
def evaluate_transcription(original, corrected):
    return {
        "WER": wer(original, corrected),
        "WIL": wil(original, corrected),
        "MER": mer(original, corrected),
    }


# Process dataset
data_dir = r"/Users/arthurzhao/Documents/CS390/SpeechData/edacc_v1.0/data/"
dataset = load_dataset(data_dir)

for path in dataset["train"]["audio"]:
    audio_path = path["path"]
    transcription, tokens = transcribe_audio(audio_path)
    uncertain_text = mask_uncertain_tokens(transcription, tokens)
    corrected_text = predict_masks(uncertain_text)
    evaluation = evaluate_transcription(transcription, corrected_text)
    print(evaluation)
