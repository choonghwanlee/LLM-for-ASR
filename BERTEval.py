from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BertTokenizer,
    BertForMaskedLM,
)
import torch
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
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    return device, whisper_model, whisper_processor, bert_tokenizer, bert_model


def process_and_predict(
    data, whisper_model, whisper_processor, bert_tokenizer, bert_model, device
):
    predictions = []
    references = []

    for batch in data:
        input_features = whisper_processor(
            batch["audio"]["array"], return_tensors="pt", sampling_rate=16000
        )
        output = whisper_model.generate(
            **input_features.to(device),
            output_scores=True,
            return_dict_in_generate=True,
        )
        transcriptions = whisper_processor.batch_decode(
            output.sequences, skip_special_tokens=True
        )

        for transcription, scores in zip(transcriptions, output.scores):
            tokens = whisper_processor.tokenizer.tokenize(transcription)
            uncertain_tokens = [
                i for i, score in enumerate(scores) if score.max().item() < 0.5
            ]
            masked_transcription = transcription

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
