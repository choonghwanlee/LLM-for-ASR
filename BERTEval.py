from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    BertTokenizer,
    BertForMaskedLM,
    WhisperConfig,
)
import torch
import re
import jiwer
import werpy
import inflect


def init_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-small"
    ).to(device)
    whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertForMaskedLM.from_pretrained("bert-base-uncased").to(device)
    whisper_model.generation_config.output_logits = True
    bert_model.generation_config.output_logits = True
    config = WhisperConfig.from_pretrained("openai/whisper-small")
    config.return_dict_in_generate = True
    return device, whisper_model, whisper_processor, bert_tokenizer, bert_model


# Standardization
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
    input = " ".join([" ".join(sublist) for sublist in input])
    return input


def filter_function(sample):
    return 15 <= len(sample["text"].split()) <= 30
    # return 50 <= len(sample["text"].split()) <= 100
    # return 15 <= len(sample["text"].split()) <= 100


def replace_numbers_with_words(text):
    p = inflect.engine()

    def num_to_words(match):
        number = match.group(0)
        return p.number_to_words(number)

    result = re.sub(r"\b\d+\b", num_to_words, text)
    return result


def join_words(words):
    sentence = ""
    for word in words:
        if word in {",", ".", "!", "?", ":", ";", "'"}:
            sentence += word
        else:
            if sentence and not sentence.endswith(" "):
                sentence += " "
            sentence += word
    return sentence


def load_and_filter_data():
    dataset = load_dataset("edinburghcstr/edacc")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
    filtered_dataset = dataset.filter(filter_function)
    sampled_dataset = (
        filtered_dataset["test"].shuffle().select(range(10))
    )  # Select a smaller sample for testing
    return sampled_dataset


def process_and_predict(
    data, whisper_model, whisper_processor, bert_tokenizer, bert_model, device
):
    predictions = []
    references = []

    for batch in data:
        inputs = whisper_processor(
            batch["audio"]["array"], return_tensors="pt", sampling_rate=16000
        )
        input_features = inputs.input_features.to(device)

        output = whisper_model.generate(
            input_features,
            task="transcribe",
            language="english",
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )

        transcription = whisper_processor.batch_decode(
            output.sequences, skip_special_tokens=True
        )[0]
        transcription = normalize(transcription)
        transcription = replace_numbers_with_words(transcription)
        print("Original transcription:", transcription)

        tokens = whisper_processor.tokenizer.tokenize(transcription)
        tokens = [token.replace("Ġ", "") for token in tokens]

        print("Tokens:", tokens)

        scores = [
            torch.softmax(output.scores[i], dim=-1).max(dim=-1).values.cpu().numpy()[0]
            for i in range(len(tokens))
        ]

        print("Scores:", scores)

        uncertain_tokens = [i for i, score in enumerate(scores) if score < 0.5]
        for idx in uncertain_tokens:
            tokens[idx] = "[MASK]"
        masked_transcription = join_words(tokens)
        print("Masked transcription:", masked_transcription)

        bert_input = bert_tokenizer(masked_transcription, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            predictions_bert = bert_model(**bert_input).logits

        for idx in (bert_input.input_ids == bert_tokenizer.mask_token_id)[0].nonzero(
            as_tuple=True
        )[0]:
            idx = idx.item() - 1
            predicted_token_id = predictions_bert[0, idx + 1].argmax(axis=-1)
            tokens[idx] = bert_tokenizer.decode(predicted_token_id).replace(" ", "")
        corrected_transcription = join_words(tokens)
        print("Corrected transcription:", corrected_transcription)

        print("Reference: ", normalize(batch["text"]))

        predictions.append(normalize(corrected_transcription))
        references.append(normalize(batch["text"]))

    return predictions, references


def compute_metrics(predictions, references):
    return {
        "WER": jiwer.wer(references, predictions) * 100,
        "MER": jiwer.mer(references, predictions) * 100,
        "WIL": jiwer.wil(references, predictions) * 100,
    }


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
