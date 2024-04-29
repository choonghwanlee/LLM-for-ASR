from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
import jiwer
import werpy
import re

def init_models(device):
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    return whisper_model, whisper_processor, bert_tokenizer, bert_model

def _process_logits(logits_list):
    max_prob_per_token = []
    for logits in logits_list:
        # Apply softmax to convert logits to probabilities for each token
        normalized_probs = F.softmax(logits, dim=-1)
        # Get the maximum probability for each token
        max_probs = torch.max(normalized_probs, dim=1)[0]  # Ensure max is taken across the correct dimension
        max_prob_per_token.extend(max_probs.tolist())
    return max_prob_per_token

def process_and_predict(batch, whisper_model, whisper_processor, bert_tokenizer, bert_model, device, gamma=0.5, threshold=0.5):
    inputs = whisper_processor(batch["audio"]["array"], return_tensors="pt", sampling_rate=16000)
    input_features = inputs.input_features.to(device)

    whisper_output = whisper_model.generate(input_features, output_scores=True, return_dict_in_generate=True)
    transcriptions = whisper_processor.batch_decode(whisper_output.sequences, skip_special_tokens=True)
    logits_list = whisper_output.scores  # This assumes scores are a list of logits tensors
    whisper_token_probs = _process_logits(logits_list)

    predictions = []
    for idx, transcription in enumerate(transcriptions):
        tokens = bert_tokenizer.tokenize(transcription)
        masked_input = bert_tokenizer(transcription, return_tensors="pt").to(device)
        bert_output = bert_model(**masked_input)
        bert_scores = bert_output.logits.softmax(dim=-1)

        combined_tokens = []
        combined_token_probs = []
        whisper_uncertain = []
        bert_uncertain = []

        # Ensure we do not exceed the bounds of available token probabilities
        num_tokens = min(len(tokens), len(whisper_token_probs))
        for token_idx in range(num_tokens):
            whisper_max_prob = whisper_token_probs[token_idx]
            bert_prob = bert_scores[0, token_idx].max().item()

            if whisper_max_prob < threshold:
                # Mix probabilities from Whisper and BERT using the gamma factor
                combined_prob = (1 - gamma) * torch.full_like(bert_scores[0, token_idx], whisper_max_prob) + gamma * bert_scores[0, token_idx]
                best_token_id = combined_prob.argmax()
                best_combined_prob = combined_prob.max().item()
                best_token = bert_tokenizer.decode([best_token_id])
                whisper_uncertain.append(tokens[token_idx])
                bert_uncertain.append(best_token)
            else:
                best_token = tokens[token_idx]
                best_combined_prob = whisper_max_prob

            combined_tokens.append(best_token)
            combined_token_probs.append(best_combined_prob)

        corrected_transcription = ' '.join(combined_tokens)
        predictions.append({
            "original_transcription": transcription,
            "corrected_transcription": corrected_transcription,
            "whisper_token_probs": whisper_token_probs[:num_tokens],  # Adjust to actual number used
            "combined_token_probs": combined_token_probs
        })

    return predictions

# Compute metrics
def compute_metrics(predictions, references):
    transformation = jiwer.Compose([
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.ExpandCommonEnglishContractions(),
    ])
    
    processed_predictions = [transformation(p) for p in predictions if p is not None]
    processed_references = [transformation(r) for r in references if r is not None]

    min_length = min(len(processed_predictions), len(processed_references))
    processed_predictions = processed_predictions[:min_length]
    processed_references = processed_references[:min_length]

    wer = jiwer.wer(processed_references, processed_predictions)
    mer = jiwer.mer(processed_references, processed_predictions)
    wil = jiwer.wil(processed_references, processed_predictions)

    return {"wer": wer, "mer": mer, "wil": wil}

# Load and preprocess data
def load_and_preprocess_data():
    dataset = load_dataset("edinburghcstr/edacc")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def filter_function(sample):
        ground_truth = sample['text']
        return 15 <= len(ground_truth.split()) <= 100

    filtered_dataset = dataset.filter(filter_function)
    return filtered_dataset['test']

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

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model, whisper_processor, bert_tokenizer, bert_model = init_models(device)
    test_dataset = load_and_preprocess_data()

    wer_results = {}
    mer_results = {}
    wil_results = {}

    for i, batch in enumerate(test_dataset):
        if i < 10:
            results = process_and_predict(batch, whisper_model, whisper_processor, bert_tokenizer, bert_model, device, gamma=0.5, threshold=0.5)
            for result in results:
                corrected_transcription = normalize(result["corrected_transcription"])
                reference_text = normalize(batch["text"])
                
                reference_length = len(reference_text.split())

                wer_score = jiwer.wer(reference_text, corrected_transcription)
                mer_score = jiwer.mer(reference_text, corrected_transcription)
                wil_score = jiwer.wil(reference_text, corrected_transcription)

                if 15 <= reference_length <= 30:
                    if "15-30 words" not in wer_results:
                        wer_results["15-30 words"] = []
                        mer_results["15-30 words"] = []
                        wil_results["15-30 words"] = []
                    wer_results["15-30 words"].append(wer_score)
                    mer_results["15-30 words"].append(mer_score)
                    wil_results["15-30 words"].append(wil_score)
                elif 50 <= reference_length <= 100:
                    if "50-100 words" not in wer_results:
                        wer_results["50-100 words"] = []
                        mer_results["50-100 words"] = []
                        wil_results["50-100 words"] = []
                    wer_results["50-100 words"].append(wer_score)
                    mer_results["50-100 words"].append(mer_score)
                    wil_results["50-100 words"].append(wil_score)
                # For the combined range
                if "15-100 words" not in wer_results:
                    wer_results["15-100 words"] = []
                    mer_results["15-100 words"] = []
                    wil_results["15-100 words"] = []
                wer_results["15-100 words"].append(wer_score)
                mer_results["15-100 words"].append(mer_score)
                wil_results["15-100 words"].append(wil_score)

    # Calculate and print average scores for each text length range
    for key in wer_results.keys():
        wer_avg = sum(wer_results[key]) / len(wer_results[key])
        mer_avg = sum(mer_results[key]) / len(mer_results[key])
        wil_avg = sum(wil_results[key]) / len(wil_results[key])

        print(f"Reference Text Length: {key}")
        print(f"WER: {wer_avg * 100:.2f}%")
        print(f"MER: {mer_avg * 100:.2f}%")
        print(f"WIL: {wil_avg * 100:.2f}%")
