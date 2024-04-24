import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from whisper import load_model

class HybridRescorer:
    def __init__(self, bert_model_name='bert-base-uncased', whisper_model_type='base'):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
        self.whisper_model = load_model(whisper_model_type)

    def predict_with_whisper(self, audio_path):
        result = self.whisper_model.transcribe(audio_path)
        return result['text']

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