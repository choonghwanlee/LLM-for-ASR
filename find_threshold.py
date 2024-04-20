from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import torchaudio

# Load the dataset and get the first audio sample
dataset = load_dataset("audiofolder", data_dir="../Project/DS_10283_4836/edacc_v1.0/data", drop_labels=True, split = "train")
audio_sample = dataset[0]["audio"]

# Load the Whisper model and processor
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")

# Resample the audio data to 16000 Hz
resampler = torchaudio.transforms.Resample(audio_sample["sampling_rate"], 16000)
audio_tensor = torch.from_numpy(audio_sample["array"]).unsqueeze(0)
resampled_audio = resampler(audio_tensor).squeeze(0)

# Preprocess the resampled audio data
audio_input = processor(
    resampled_audio,
    return_tensors="pt"
).input_features

# Move the input tensor to GPU if available
if torch.cuda.is_available():
    audio_input = audio_input.to("cuda")

# Generate the transcription
transcription_output = model.generate(audio_input)

# Decode the transcription output
transcription_text = processor.batch_decode(transcription_output, skip_special_tokens=True)[0]

# Print the transcription
print(transcription_text)

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