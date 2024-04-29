from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BertTokenizer,
    BertForMaskedLM,
)
import torch
import torch.nn.functional as F
import re
import jiwer


def filter_function(sample):
    return 15 <= len(sample["text"].split()) <= 100


def contains_number(text):
    return bool(re.search(r"\d", text))


def load_and_filter_data():
    dataset = load_dataset("edinburghcstr/edacc")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    filtered_dataset = dataset.filter(filter_function)
    sampled_dataset = (
        filtered_dataset["test"].shuffle().select(range(10))
    )  # Select a smaller sample for testing
    return sampled_dataset


def init_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    ).to(device)
    whisper_model.generation_config.output_logits = True
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    return device, whisper_model, whisper_processor, bert_tokenizer, bert_model


def process_and_predict(
    data, whisper_model, whisper_processor, bert_tokenizer, bert_model, device
):
    predictions = []
    references = []

    for batch in data: ## loop over each audio sample in data
        input_features = whisper_processor(
            batch["audio"]["array"], return_tensors="pt", sampling_rate=16000
        ).input_features.to(device)
        output = whisper_model.generate(
            input_features,
            generation_config = whisper_model.generation_config, 
            task='transcribe', 
            language='english',
            # output_scores=True, ## you can't actually edit this directly 
            return_dict_in_generate=True,
        )
        transcriptions = whisper_processor.batch_decode(
            output.sequences, skip_special_tokens=True
        ) 
        tokens = output.sequences ## how to associate tokens
        normalized_probs = [F.softmax(logit) for logit in output['scores']] ## list of lists of normalized probs distributions per token
        max_prob_per_token = [torch.max(probs).item() for probs in normalized_probs] ## list of maximum probabilities per token
        uncertain_tokens = [index for index, prob in enumerate(max_prob_per_token) if prob < 0.5] ## indexes of tokens with max_prob < 0.5
        ## for each uncertain token, replace it with [MASK], otherwise decode ##  
        # ["[MASK]" for i, token in  range(len(tokens)) if i in uncertain_tokens else whisper_processor.decode(token)]
        # masked_transcription = " ".join(transcript_as_list)
        bert_input = bert_tokenizer(masked_transcription, return_tensors="pt")
        masked_indices = torch.where(
            bert_input["input_ids"] == bert_tokenizer.mask_token_id
        )[1]
        predictions_bert = bert_model(**bert_input.to(device)).logits
        predicted_token_id = predictions_bert[0, idx].argmax(axis=-1)


        for index, (token, probs) in enumerate(zip(tokens, normalized_probs)): ## individaul decoded words and corresponding probability 
            print(token)
            print(probs)
            uncertain_tokens = [
                i for i, score in enumerate(probs) if score.max().item() < 0.5
            ] ## get indices of uncertain toen
            masked_transcription = transcriptions
            for idx in uncertain_tokens:
                tokens[idx] = "[MASK]"
            masked_transcription = " ".join(tokens)

            bert_input = bert_tokenizer(masked_transcription, return_tensors="pt")
            masked_indices = torch.where(
                bert_input["input_ids"] == bert_tokenizer.mask_token_id
            )[1]
            predictions_bert = bert_model(**bert_input.to(device)).logits

            for idx in masked_indices:
                predicted_token_id = predictions_bert[0, idx].argmax(axis=-1)
                tokens[idx] = bert_tokenizer.decode([predicted_token_id])

            corrected_transcription = " ".join(tokens)
            predictions.append(corrected_transcription)
            references.append(batch["text"])

    return predictions, references


def compute_metrics(predictions, references):
    return {
        "WER": jiwer.wer(references, predictions) * 100,
        "MER": jiwer.mer(references, predictions) * 100,
        "WIL": jiwer.wil(references, predictions) * 100,
    }


# Main function
def main():
    data = load_and_filter_data()
    device, whisper_model, whisper_processor, bert_tokenizer, bert_model = init_models()
    predictions, references = process_and_predict(
        data, whisper_model, whisper_processor, bert_tokenizer, bert_model, device
    )
    metrics = compute_metrics(predictions, references)
    print(metrics)


if __name__ == "__main__":
    main()
