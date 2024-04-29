from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BertTokenizer,
    BertForMaskedLM,
)
import torch
import torch.nn.functional as F

def init_models(device):
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    return whisper_model, whisper_processor, bert_tokenizer, bert_model

def process_and_predict(batch, whisper_model, whisper_processor, bert_tokenizer, bert_model, device, gamma=0.5, threshold=0.5):
    inputs = whisper_processor(batch["audio"]["array"], return_tensors="pt", sampling_rate=16000)
    input_features = inputs.input_features.to(device)

    # Generate predictions and extract logits
    whisper_output = whisper_model.generate(input_features, output_scores=True, return_dict_in_generate=True)
    transcriptions = whisper_processor.batch_decode(whisper_output.sequences, skip_special_tokens=True)


    logits = whisper_output.scores[0]  # Extract logits from the output

    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits, dim=-1)

    print("Probabilities:", probabilities)
    print("Sum of probabilities:", torch.sum(probabilities))


    predictions = []
    for idx, (transcription, prob) in enumerate(zip(transcriptions, probabilities)):
        tokens = bert_tokenizer.tokenize(transcription)
        masked_input = bert_tokenizer(transcription, return_tensors="pt").to(device)
        
        # Get BERT predictions
        bert_output = bert_model(**masked_input)
        bert_scores = bert_output.logits.softmax(dim=-1)
        
        combined_tokens = []
        whisper_token_probs = []
        combined_token_probs = []

        for token_idx, token in enumerate(tokens):
            # Extract the maximum probability for the predicted token
            whisper_max_prob = prob[token_idx].max().item()
            bert_prob = bert_scores[0, token_idx].max().item()  # Correctly define bert_prob

            whisper_token_probs.append(whisper_max_prob)

            if whisper_max_prob < threshold:
                # Apply hybrid rescoring only if below threshold
                combined_prob = (1 - gamma) * prob[token_idx] + gamma * bert_scores[0, token_idx]  # Use full bert_scores for that token
                best_token_id = combined_prob.argmax()
                best_combined_prob = combined_prob.max().item()
            else:
                best_token_id = prob[token_idx].argmax()
                best_combined_prob = whisper_max_prob

            best_token = bert_tokenizer.decode([best_token_id])
            combined_tokens.append(best_token)
            combined_token_probs.append(best_combined_prob)

        corrected_transcription = ' '.join(combined_tokens)
        predictions.append({
            "original_transcription": transcription,
            "corrected_transcription": corrected_transcription,
            "whisper_token_probs": whisper_token_probs,
            "combined_token_probs": combined_token_probs
        })

    return predictions

 
def load_and_preprocess_data():
    # Load the dataset
    dataset = load_dataset("edinburghcstr/edacc")
    
    # Cast the 'audio' column to Audio format with the desired sample rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Filter the dataset to include only entries with text lengths within a specific range
    def filter_function(sample):
        return 15 <= len(sample['text'].split()) <= 100  # Adjust this range as needed

    # Apply the filter function
    filtered_dataset = dataset.filter(filter_function)
    
    return filtered_dataset['test']  # Assuming you want to work with the test split for predictions

# Running the entire script
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model, whisper_processor, bert_tokenizer, bert_model = init_models(device)
    test_dataset = load_and_preprocess_data()

    # Process each batch in the dataset
    for i, batch in enumerate(test_dataset):
        if i < 10:  # Process only the first 10 samples for demonstration
            results = process_and_predict(batch, whisper_model, whisper_processor, bert_tokenizer, bert_model, device, gamma=0.5, threshold=0.5)
            for result in results:
                print(f"Original: {result['original_transcription']}")
                print(f"Corrected: {result['corrected_transcription']}")
                print("Whisper Probabilities:", result['whisper_token_probs'])
                print("Hybrid Probabilities:", result['combined_token_probs'])
                print("\n")