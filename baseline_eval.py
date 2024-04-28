## Baseline evaluation of Whisper model 
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy 
import pandas
import jiwer
import werpy
from evaluate import load


# Step 1: Load the dataset
dataset = load_dataset('edinburghcstr/edacc') ## install from HF instead
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) ## resample to 16 kHz

## sample usage
sample = dataset["test"] ## sample a row from the validation dataset
audio_sample = sample['audio'] ## audio feature
ground_truth = sample['text'] ## ground truth transcription
accent = sample['accent'] ## speaker accent 

# Filter the dataset to include only samples with ground truth text longer than 10 words
def filter_function(sample):
    ground_truth = werpy.normalize(sample['text'])
    return len(ground_truth.split()) > 10

filtered_dataset = dataset.filter(filter_function)

# Step 2: Load the Whisper model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

## change configs to allow logits and scores
model.generation_config.output_logits = True

candidates = []
references = [] 

# Step 3: Preprocess the resampled audio data
for sample in filtered_dataset["test"]:
    audio_sample = sample['audio']
    ground_truth = sample['text']
    waveform = audio_sample["array"]
    sampling_rate = audio_sample["sampling_rate"]
    audio_input = processor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features
    audio_input = audio_input.to(device)

    # Step 4: Generate the transcription
    with torch.no_grad():
        output = model.generate(input_features=audio_input, generation_config=model.generation_config, task='transcribe', language='english', return_dict_in_generate=True)

    # Get the actual prediction
    transcription = processor.batch_decode(output['sequences'], skip_special_tokens=True)[0]
    candidate = werpy.normalize(transcription)
    candidates.append(candidate)
    reference = werpy.normalize(ground_truth)
    references.append(reference)

print(len(candidates))
print(len(references))

# wer = load("wer") 
# wer_score = wer.compute(predictions=ds["predictions"], references=ds["references"])

# print(f"WER: {wer_score * 100:.2f} %")
