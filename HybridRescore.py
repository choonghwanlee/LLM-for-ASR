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

def load_and_preprocess_data():
    dataset = load_dataset("edinburghcstr/edacc")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Optionally filter the dataset if required
    def filter_function(sample):
        return 15 <= len(sample['text'].split()) <= 100  # Adjust the range as needed

    filtered_dataset = dataset.filter(filter_function)
    return filtered_dataset['test']  # Assuming we are using the test split

def process_and_predict(batch, whisper_model, whisper_processor, bert_tokenizer, bert_model, device, gamma=0.5, threshold=0.5):
    inputs = whisper_processor(batch["audio"]["array"], return_tensors="pt", sampling_rate=16000)
    input_features = inputs.input_features.to(device)

    whisper_output = whisper_model.generate(input_features, output_scores=True, return_dict_in_generate=True)
    transcriptions = whisper_processor.batch_decode(whisper_output.sequences, skip_special_tokens=True)
    whisper_scores = whisper_output.scores[0].softmax(dim=-1)  # Softmax over logits

    predictions = []
    for idx, (transcription, scores) in enumerate(zip(transcriptions, whisper_scores)):
        tokens = bert_tokenizer.tokenize(transcription)
        masked_input = bert_tokenizer(transcription, return_tensors="pt").to(device)
        
        bert_output = bert_model(**masked_input)
        bert_scores = bert_output.logits.softmax(dim=-1)
        
        combined_tokens = []
        for token_idx, token in enumerate(tokens):
            whisper_prob = scores[token_idx]
            bert_prob = bert_scores[0, token_idx]

            if whisper_prob.max().item() < threshold:
                combined_prob = (1 - gamma) * whisper_prob + gamma * bert_prob
                best_token_id = combined_prob.argmax()
                best_token = bert_tokenizer.decode([best_token_id])
                combined_tokens.append(best_token)
            else:
                best_token = bert_tokenizer.decode([whisper_prob.argmax().item()])
                combined_tokens.append(best_token)
        
        corrected_transcription = ' '.join(combined_tokens)
        predictions.append(corrected_transcription)

    return predictions

# Main function
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model, whisper_processor, bert_tokenizer, bert_model = init_models(device)
    test_dataset = load_and_preprocess_data()  # Load and preprocess the data

    # Process each batch in the dataset
    for i, batch in enumerate(test_dataset):
        if i < 10:  # Process only the first 10 samples for demonstration
            predictions = process_and_predict(batch, whisper_model, whisper_processor, bert_tokenizer, bert_model, device, gamma=0.5, threshold=0.5)
            print(f"Original: {batch['text']}\nCorrected: {predictions[0]}\n")
