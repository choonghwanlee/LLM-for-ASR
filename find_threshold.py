## Find threshold tau that determines whether a prediction is certain or uncertain

import torch
import numpy 
import matplotlib.pyplot as plt
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

# pipeline example
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

dataset = load_dataset("/Users/rebekahkim/Desktop/dataset/edacc_v1.0/data")
result = pipe("/Users/rebekahkim/Desktop/dataset/edacc_v1.0/data/EACC-CO1.wav")

print(result["text"])



'''
# run the Whisper model on all the data
predictions = [pipe(audio_file) for audio_file in dataset]

# calculate the maximum predicted probabilities for each token
max_probs = [numpy.max(prediction['score']) for prediction in predictions]

# plot the distribution of the maximum predicted probabilities
plt.hist(max_probs)
plt.title('Distribution of Maximum Predicted Probabilities')
plt.xlabel('Probability')
plt.ylabel('Frequency')
plt.show()
'''