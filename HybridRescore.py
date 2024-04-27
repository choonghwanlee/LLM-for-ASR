import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification, WhisperForConditionalGeneration, WhisperProcessor
import torch.nn.functional as F
# from whisper import load_model

class HybridRescorer:
    def __init__(self, bert_model_name='bert-base-uncased', whisper_model_type='base'):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_type)
        self.whisper_model.generation_config.output_logits = True
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_type)
        # self.whisper_model = load_model(whisper_model_type)

    def predict_with_whisper(self, waveform, sampling_rate):
        audio_input = self.whisper_processor(
            waveform,
            sampling_rate = sampling_rate,
            return_tensors="pt"
        ).input_features
        result = self.whisper_model.generate(audio_input, generation_config= self.whisper_model.generation_config, task='transcribe', language='english')
        transcription = self.whisper_processor.batch_decode(result['sequences'], skip_special_tokens=True)[0]
        normalized_probs = [F.softmax(logit) for logit in result['logits']]
        max_prob_per_token = [torch.max(probs).item() for probs in normalized_probs]
        return transcription, max_prob_per_token

    def score_with_bert(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probabilities[:, 1].item()  

    def hybrid_rescore(self, audio_path):
        whisper_prediction = self.predict_with_whisper(audio_path)
        bert_score = self.score_with_bert(whisper_prediction)
        return whisper_prediction, bert_score

def process_corpus_audio_files(data_dir):
    rescorer = HybridRescorer()
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):  
                audio_path = os.path.join(root, file)
                prediction, score = rescorer.hybrid_rescore(audio_path)
                print(f"Audio File: {audio_path}")
                print(f"Whisper Prediction: {prediction}, BERT Score: {score}\n")

if __name__ == "__main__":
    data_dir = "C:\\Users\\divya\Downloads\\DS_10283_4836\\edacc_v1.0\\edacc_v1.0\\data"  # Adjust this path to your dataset location
    process_corpus_audio_files(data_dir)