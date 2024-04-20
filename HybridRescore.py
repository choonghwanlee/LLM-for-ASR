## Approach 1: using a hybrid rescoring between Whisper predictions and BERT predictions

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
            return probabilities[:, 1].item()  # Assuming index 1 is the target class

    def hybrid_rescore(self, audio_path):
        whisper_prediction = self.predict_with_whisper(audio_path)
        bert_score = self.score_with_bert(whisper_prediction)
        return whisper_prediction, bert_score

# if __name__ == "__main__":
#     rescorer = HybridRescorer()
#     audio_path = "path/to/your/audio/file.wav"
#     prediction, score = rescorer.hybrid_rescore(audio_path)
#     print(f"Whisper Prediction: {prediction}, BERT Score: {score}")