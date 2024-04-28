## Baseline evaluation of Whisper model 
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import jiwer
import werpy
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
dataset = load_dataset('edinburghcstr/edacc') ## install from HF instead
dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) ## resample to 16 kHz

# Filter the dataset to include only samples with ground truth text longer than 10 words
def filter_function(sample):
    ground_truth = sample['text']
    return 15 <= len(ground_truth.split()) <= 100

filtered_dataset = dataset.filter(filter_function)
# check length 
print(len(filtered_dataset["test"]))
# print(filtered_dataset["test"][0])

# Step 2: Load the Whisper model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

## change configs to allow logits and scores
model.generation_config.output_logits = True

def contains_number(text):
    return bool(re.search(r'\d', text))

def map_fn(batch):
    arrays = [x["array"] for x in batch["audio"]]
    sampling_rate = [x['sampling_rate'] for x in batch['audio']]
    input_features = processor.feature_extractor(arrays, sampling_rate=sampling_rate[0], return_tensors="pt").input_features.to(device)
    sequences = model.generate(input_features, task='transcribe', language='english', use_cache=True)
    results = processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)
    
    # Check each prediction for numbers
    predictions = []
    references = []
    for result, reference in zip(results, batch["text"]):
        if not contains_number(result):
            predictions.append(normalize(result))
            references.append(normalize(reference))
        else:
            # Append placeholder values for filtered samples
            predictions.append(None)
            references.append(None)
    
    return {"predictions": predictions, "references": references}

ds = filtered_dataset["test"].map(map_fn, batch_size=4, remove_columns=[], batched=True) ## use a batch size of 4


# Filter function to remove samples with None values
def filter_none_samples(example):
    # Check if both predictions and references are not None
    return example["predictions"] is not None and example["references"] is not None

# Filter the dataset
filtered_ds = ds.filter(filter_none_samples)

# Now calculate the WER
predictions = [example["predictions"] for example in filtered_ds]
references = [example["references"] for example in filtered_ds]

wer = load("wer")
mer = load("mer")
wil = load("wil")
wer_score = wer.compute(predictions=predictions, references=references)
mer_score = mer.compute(predictions=predictions, references=references)
wil_score = wil.compute(predictions=predictions, references=references)
print(f"WER: {wer_score * 100:.2f} %")
print(f"MER: {mer_score * 100:.2f} %")
print(f"WIL: {wil_score * 100:.2f} %")
