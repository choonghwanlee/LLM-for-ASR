import torch
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BertTokenizer, BertForMaskedLM
import werpy
import jiwer
import torch.nn.functional as F
# from evaluate import load
import re
from utils import get_punctuation_tokens


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
    to_modify= output['sequences']
    logits = output['logits']
    normalized_probs = [F.softmax(logit) for logit in logits]
    max_prob_per_token = [torch.max(probs).item() for probs in normalized_probs] 
    # transcription = processor.batch_decode(output['sequences'], skip_special_tokens=True)[0]
    uncertain_tokens = [index for index, prob in enumerate(max_prob_per_token) if prob < uncertainty_threshold and to_modify[0][index] not in get_punctuation_tokens()] 
    whisper_top_k = []
    for i, index in enumerate(uncertain_tokens): ## for each uncertain token
        top_k_index = torch.topk(normalized_probs[i], k=top_n_predictions).indices
        whisper_top_k.append([processor.decode(val) for val in top_k_index[0]]) ## find the probability and token id of top 3
        to_modify[0][index] = 50360
    raw_transcript = processor.batch_decode(to_modify, skip_special_tokens=False)[0]
    to_remove = ["<|startoftranscript|>", "<|translate|>", "<|en|>", "<|transcribe|>", "<|notimestamps|>", "<|endoftext|>", "<|startoflm|>"]
    for substring in to_remove: 
        if substring == "<|startoflm|>":
            raw_transcript = raw_transcript.replace(substring, " [MASK]")
        else:
            raw_transcript = raw_transcript.replace(substring, "")

    return raw_transcript, uncertain_tokens, whisper_top_k

def predict_masks(masked_text, uncertain_indices, k_best):
    whisper_k_best = []
    for mask in k_best:
        print('hi!')
        whisper_k_best.append([val[1].item() for val in tokenizer_bert(mask, return_tensors="pt").input_ids])
    print(whisper_k_best)
    # whisper_top_k = [val[1].item() for val in tokenizer_bert(k_best, return_tensors="pt").input_ids]
    bert_inputs = tokenizer_bert(masked_text, return_tensors="pt")
    with torch.no_grad():
        logits = model_bert(**bert_inputs).logits
    # normalize values
    normalized = [F.softmax(logit) for logit in logits]
    input_tokens = bert_inputs.input_ids 
    mask_token_index = (input_tokens == tokenizer_bert.mask_token_id)
    # Find indices where values are True
    mask_token_index = torch.nonzero(mask_token_index.flatten()).flatten()
    bert_pred = []
    for i, mask_idx in enumerate(mask_token_index): ## for each [MASK] token
        candidates = [normalized[0][mask_idx][index] for index in whisper_k_best[i]] ## find the normalized probability of Whisper's top K
        most_likely = whisper_k_best[i][candidates.index(max(candidates))] ## find the most likely token among the top K
        input_tokens[0][mask_idx] = most_likely
    return tokenizer_bert.batch_decode(input_tokens)
    # predicted_tokens = []
    # for idx, uncertain_idx in enumerate(uncertain_indices):
    #     mask_logits = logits[0, uncertain_idx]
    #     top_n_logits, top_n_ids = torch.topk(mask_logits, top_n, dim=-1)
    #     predicted_tokens.append([tokenizer_bert.convert_ids_to_tokens([id])[0] for id in top_n_ids.tolist()])
        
    # return predicted_tokens

# def replace_uncertain_tokens(original_text, uncertain_indices, predicted_tokens):
#     words = original_text.split()
#     for i, idx in enumerate(uncertain_indices):
#         words[idx] = predicted_tokens[i][0]  # Replace with the token with highest logit value
#     return " ".join(words)

wer_scores = []
wil_scores = []
mer_scores = []
num_words = []

for sample in filtered_dataset["test"]:
    audio_sample = sample['audio']  # audio feature
    ground_truth = sample['text']  # ground truth transcription
    accent = sample['accent']  # speaker accent 
    
    # Extract uncertain tokens and masked BERT input
    transcription, uncertain_indices, k_best = extract_uncertain(audio_sample)
    
    # Predict top-N tokens using BERT MLM
    new_transcription = predict_masks(transcription, uncertain_indices, k_best)
    
    # Replace uncertain tokens with predicted tokens
    # corrected_text = replace_uncertain_tokens(transcription, uncertain_indices, predicted_tokens)
    normalized_ground_truth = normalize(ground_truth)
    normalized_corrected_text = normalize(new_transcription)
    
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

