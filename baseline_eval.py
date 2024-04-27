## Baseline evaluation of Whisper model 
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import torchaudio
import matplotlib.pyplot as plt
# from evaluate import load


# Step 1: Load the dataset
dataset = load_dataset('edinburghcstr/edacc') ## install from HF instead
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) ## resample to 16 kHz

## sample usage
sample = dataset["validation"][0] ## sample a row from the validation dataset
audio_sample = sample['audio'] ## audio feature
ground_truth = sample['text'] ## ground truth transcription
accent = sample['accent'] ## speaker accent

# Step 2: Load the Whisper model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

## batch prediction – move to a separate eval script:
def map_fn(batch):
    arrays = [x["array"] for x in batch["audio"]]
    sampling_rate = [x['sampling_rate'] for x in batch['audio']]
    input_features = processor.feature_extractor(arrays, sampling_rate=sampling_rate[0], return_tensors="pt").input_features.to(device)
    sequences = model.generate(input_features, task='transcribe', language='english', use_cache=True)
    results = processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)
    batch["predictions"] = [result for result in results]
    batch["references"] = [processor.tokenizer._normalize(text) for text in batch["text"]]
    return batch

ds = dataset['validation'].map(map_fn, batch_size=4, remove_columns=[], batched=True) ## use a batch size of 4

# wer = load("wer") 
# wer_score = wer.compute(predictions=ds["predictions"], references=ds["references"])

# print(f"WER: {wer_score * 100:.2f} %")
