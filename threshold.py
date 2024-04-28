from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperFeatureExtractor
import torch
import torchaudio
from evaluate import load


print('hi!')
# Step 1: Load the dataset
# dataset = load_dataset("audiofolder", data_dir="./dataset/edacc_v1.0/data", drop_labels=True, split = "train")
dataset = load_dataset('edinburghcstr/edacc') ## install from HF instead

dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)) ## resample to 16 kHz

## sample usage
sample = dataset["validation"][0] ## sample a row from the validation dataset
audio_sample = sample['audio'] ## audio feature
ground_truth = sample['text'] ## ground truth transcription
accent = sample['accent'] ## speaker accent

# Step 2: Load the Whisper model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

device = "cuda" if torch.cuda.is_available() else "cpu"


# # Resample the audio data to 16000 Hz
# resampler = torchaudio.transforms.Resample(audio_sample["sampling_rate"], 16000)
# audio_tensor = torch.from_numpy(audio_sample["array"]).unsqueeze(0)
# resampled_audio = resampler(audio_tensor).squeeze(0)

# Step 4: Preprocess the audio data
waveform = audio_sample["array"]
sampling_rate = audio_sample["sampling_rate"]
input_features = feature_extractor(waveform, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

# Step 5: Generate the transcription with torch.no_grad():
decoder_input_ids = processor.prepare_for_generation(input_features)
generated_ids = model.generate(decoder_input_ids, task='transcribe', language='english')
transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)

transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(transcription)

## batch prediction:
def map_fn(batch):
    arrays = [x["array"] for x in batch["audio"]]
    sampling_rate = [x['sampling_rate'] for x in batch['audio']]
    input_features = processor.feature_extractor(arrays, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)
    sequences = model.generate(input_features, task='transcribe', language='english', use_cache=True)
    results = processor.tokenizer.batch_decode(sequences, skip_special_tokens=True)
    batch["predictions"] = [result for result in results]
    batch["references"] = [processor.tokenizer._normalize(text) for text in batch["text"]]
    return batch

ds = dataset['validation'].map(map_fn, batch_size=4, remove_columns=[], batched=True) ## use a batch size of 4 

wer = load("wer")
wer_score = wer.compute(predictions=ds["predictions"], references=ds["references"])

print(f"WER: {wer_score * 100:.2f} %")



### Old Code

# import torch
# from transformers import WhisperForConditionalGeneration, WhisperProcessor
# from datasets import load_dataset
# import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
# processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# ds = load_dataset("audiofolder", data_dir="../Project/DS_10283_4836/edacc_v1.0/data", drop_labels=True, split = "train")
# audio_sample = ds[0]
# print(ds["audio"])

# audio_sample = ds[0]["audio"]
# input_features = processor(audio_sample["array"], sampling_rate=audio_sample["sampling_rate"], return_tensors="pt").input_features.to(device)
# generated_ids = model.generate(input_features)
# transcription = processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0]

# #text = audio_sample["text"].lower()
# # speech_data = audio_sample["audio"]["array"]
# # print(len(speech_data), speech_data)
# # inputs = processor.feature_extractor(speech_data, return_tensors="pt", sampling_rate=16_000).input_features.to(device)
# # inputs
# # print(inputs.shape)

# # predicted_ids = model.generate(inputs, max_length=480_000)
# # predicted_ids