from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F

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
            else:
                best_token_id = bert_scores[0, token_idx].argmax()
                best_combined_prob = whisper_max_prob

            best_token = bert_tokenizer.decode([best_token_id])
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

def load_and_preprocess_data():
    dataset = load_dataset("edinburghcstr/edacc")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    filtered_dataset = dataset.filter(lambda sample: 15 <= len(sample['text'].split()) <= 100)
    return filtered_dataset['test']

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model, whisper_processor, bert_tokenizer, bert_model = init_models(device)
    test_dataset = load_and_preprocess_data()

    for i, batch in enumerate(test_dataset):
        if i < 10:  # Process only the first 10 samples for demonstration
            results = process_and_predict(batch, whisper_model, whisper_processor, bert_tokenizer, bert_model, device, gamma=0.5, threshold=0.5)
            for result in results:
                print(f"Original: {result['original_transcription']}")
                print(f"Corrected: {result['corrected_transcription']}")
                print("Whisper Probabilities:", result['whisper_token_probs'])
                print("Hybrid Probabilities:", result['combined_token_probs'])
                print("\n")
