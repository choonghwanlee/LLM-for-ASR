from transformers import BertTokenizer, BertForSequenceClassification, WhisperForConditionalGeneration, WhisperProcessor
import torch
import torch.nn.functional as F
import torchaudio
import os
from datasets import load_dataset, Audio


class HybridRescorer:
    def __init__(self, bert_model_name='bert-base-uncased', whisper_model_type='openai/whisper-small', gamma=0.5):
        # Initialize BERT model and tokenizer
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
        
        # Initialize Whisper model and processor
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_type)
        self.processor = WhisperProcessor.from_pretrained(whisper_model_type)
        self.whisper_model.generation_config.output_logits = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model.to(self.device)
        
        # Other parameters
        self.threshold = 0.5  # temporary Threshold rn for low confidence in Whisper prediction
        self.gamma = gamma  # Weight parameter for combining Whisper and BERT probabilities that have to fine tune

    def load_audio(self, audio_path):
        waveform, _ = torchaudio.load(audio_path)
        return waveform

    def predict_with_whisper(self, audio_input):
        # Preprocess audio input
        inputs = self.processor(
            audio_input,
            sampling_rate=16000,
            return_tensors="pt"
        )
    
        # Ensure the input tensor has the expected shape
        input_features = inputs.input_features.to(self.device)

        # Generate transcription
        with torch.no_grad():
            output = self.whisper_model.generate(
                input_features=input_features,
                generation_config=self.whisper_model.generation_config,
                task='transcribe',
                language='english',
                return_dict_in_generate=True
            )
            transcription = self.processor.batch_decode(output['sequences'], skip_special_tokens=True)[0]

            # Compute max probability per token
            max_prob_per_token = self._process_logits(output['logits'])

        print("transcribe",transcription)
        print("max token",max_prob_per_token)
        return transcription, max_prob_per_token

    def _process_logits(self, logits):
        normalized_probs = [F.softmax(logit, dim=-1) for logit in logits]
        max_prob_per_token = [torch.max(probs, dim=-1).values.item() for probs in normalized_probs]
        return max_prob_per_token

    def score_with_bert(self, text):
        inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            return probabilities[:, 1].item()

    def hybrid_rescoring(self, audio_input):
        whisper_prediction, confidence = self.predict_with_whisper(audio_input)
        uncertain_tokens = [i for i, conf in enumerate(confidence) if conf < self.threshold]

        rescored_transcription = []
        for i in uncertain_tokens:
        # Ensure the index is within bounds
             if i < len(whisper_prediction.split()):
            # Probability of token being correct from Whisper
                whisper_prob = confidence[i]

                token_text = whisper_prediction.split()[i]  # Get the text of the uncertain token
                bert_prob = self.score_with_bert(token_text)  # Use BERT to predict the probability of the token

                weighted_prob = (1 - self.gamma) * whisper_prob + self.gamma * bert_prob

                rescored_transcription.append((i, weighted_prob))
             else:
                print("Index out of range for token:", i)

        return rescored_transcription


if __name__ == "__main__":
    data_dir = r"C:\Users\divya\Downloads\DS_10283_4836\edacc_v1.0\edacc_v1.0\data"
    dataset = load_dataset(data_dir)  ## Load dataset from Hugging Face

    # Initialize HybridRescorer
    rescorer = HybridRescorer()

    print("please bro",dataset.keys())

    # Iterate over the dataset
    for sample in dataset["train"]:
        
        audio_sample = sample['audio']  ## Audio feature
        print(audio_sample)
        # ground_truth = sample['text']  ## Ground truth transcription

        # Load audio and perform hybrid rescoring
        audio_input = audio_sample["array"]
        rescored_transcription = rescorer.hybrid_rescoring(audio_input)

        # Print results
        # print(f"Ground Truth: {ground_truth}")
        print(f"Rescored Transcription: {rescored_transcription}")



# from transformers import BertTokenizer, BertForSequenceClassification, WhisperForConditionalGeneration, WhisperProcessor
# import torch
# import torch.nn.functional as F
# import torchaudio
# import os
# from datasets import load_dataset, Audio


# class HybridRescorer:
#     def __init__(self, bert_model_name='bert-base-uncased', whisper_model_type='openai/whisper-small', gamma=0.5):
#         # Initialize BERT model and tokenizer
#         self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#         self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
        
#         # Initialize Whisper model and processor
#         self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_type)
#         self.processor = WhisperProcessor.from_pretrained(whisper_model_type)
#         self.whisper_model.generation_config.output_logits = True
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.whisper_model.to(self.device)
        
#         # Other parameters
#         self.threshold = 0.5  # temporary Threshold rn for low confidence in Whisper prediction
#         self.gamma = gamma  # Weight parameter for combining Whisper and BERT probabilities that have to fine tune

#     def load_audio(self, audio_path):
#         waveform, _ = torchaudio.load(audio_path)
#         return waveform

#     def predict_with_whisper(self, audio_input):
#     # Preprocess audio input
#         inputs = self.processor(
#         audio_input,
#         sampling_rate=16000,
#         return_tensors="pt"
#     )
    
#     # Ensure the input tensor has the expected shape
#         input_features = inputs.input_features.to(self.device)
#         print("inputututututu", input_features)

#     # Generate transcription
#         with torch.no_grad():
#             output = self.whisper_model.generate(
#                 input_features=input_features,
#                 generation_config=self.whisper_model.generation_config,
#                 task='transcribe',
#                 language='english',
#                 return_dict_in_generate=True
#         )
#             transcription = self.processor.batch_decode(output['sequences'], skip_special_tokens=True)[0]

#         # Compute max probability per token
#             max_prob_per_token = self._process_logits(output['logits'])

#         print(transcription)
#         print(max_prob_per_token)
#         return transcription, max_prob_per_token

#     def _process_logits(self, logits):
#         normalized_probs = [F.softmax(logit, dim=-1) for logit in logits]
#         max_prob_per_token = [torch.max(probs, dim=-1).values.item() for probs in normalized_probs]
#         return max_prob_per_token

#     def score_with_bert(self, text):
#         inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.bert_model(**inputs)
#             probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             return probabilities[:, 1].item()

#     def hybrid_rescoring(self, audio_path):
#         audio_input = self.load_audio(audio_path)
#         whisper_prediction, confidence = self.predict_with_whisper(audio_input)
#         uncertain_tokens = [i for i, conf in enumerate(confidence) if conf < self.threshold]

#         rescored_transcription = []
#         for i in uncertain_tokens:
#         # Probability of token being correct from Whisper
#             whisper_prob = confidence[i]

#             token_text = whisper_prediction.split()[i]  # Get the text of the uncertain token
#             bert_prob = self.score_with_bert(token_text)  # Use BERT to predict the probability of the token

#             weighted_prob = (1 - self.gamma) * whisper_prob + self.gamma * bert_prob

#             rescored_transcription.append((i, weighted_prob))

#         return rescored_transcription


# if __name__ == "__main__":
#     data_dir = r"C:\Users\divya\Downloads\DS_10283_4836\edacc_v1.0\edacc_v1.0\data"
#     dataset = load_dataset(data_dir) ## install from HF instead
#     print("success 1")
#     dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))
#     print("success 2")

#     rescorer = HybridRescorer()

    

#     for sample in dataset:
#         audio_sample = sample['audio']  ## audio feature
#         ground_truth = sample['text']  ## ground truth transcription
#         accent = sample['accent']  ## speaker accent

#         # Load audio and perform hybrid rescoring
#         audio_input = audio_sample["array"]
#         rescored_transcription = rescorer.hybrid_rescoring(audio_input)

#         # Print results
#         print(f"Ground Truth: {ground_truth}")
#         print(f"Rescored Transcription: {rescored_transcription}")



# from transformers import BertTokenizer, BertForSequenceClassification, WhisperForConditionalGeneration, WhisperProcessor
# import torch
# import torch.nn.functional as F
# import torchaudio
# import librosa
# import os

# class HybridRescorer:
#     def __init__(self, bert_model_name='bert-base-uncased', whisper_model_type='openai/whisper-small', gamma=0.5):
#         # Initialize BERT model and tokenizer
#         self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
#         self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
        
#         # Initialize Whisper model and processor
#         self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_type)
#         self.processor = WhisperProcessor.from_pretrained(whisper_model_type)
#         self.whisper_model.generation_config.output_logits = True
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.whisper_model.to(self.device)
        
#         # Other parameters
#         self.threshold = 0.5  # temporary Threshold rn for low confidence in Whisper prediction
#         self.gamma = gamma  # Weight parameter for combining Whisper and BERT probabilities that have to fine tune

#     def predict_with_whisper(self, audio_input):
#         # Preprocess audio input
#         input_features = self.processor(
#             audio_input,
#             sampling_rate=16000,
#             return_tensors="pt"
#         ).input_features.to(self.device)

#         # Generate transcription
#         with torch.no_grad():
#             output = self.whisper_model.generate(
#                 input_features=input_features,
#                 generation_config=self.whisper_model.generation_config,
#                 task='transcribe',
#                 language='english',
#                 return_dict_in_generate=True
#             )
#             transcription = self.processor.batch_decode(output['sequences'], skip_special_tokens=True)[0]

#             # Compute max probability per token
#             max_prob_per_token = self._process_logits(output['logits'])

#         print(transcription)
#         print(max_prob_per_token)
#         return transcription, max_prob_per_token

#     def _process_logits(self, logits):
#         normalized_probs = [F.softmax(logit, dim=-1) for logit in logits]
#         max_prob_per_token = [torch.max(probs, dim=-1).values.item() for probs in normalized_probs]
#         return max_prob_per_token

#     def score_with_bert(self, text):
#         inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.bert_model(**inputs)
#             probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
#             return probabilities[:, 1].item()

#     def hybrid_rescoring(self, audio_input):
#         whisper_prediction, confidence = self.predict_with_whisper(audio_input)
#         uncertain_tokens = [i for i, conf in enumerate(confidence) if conf < self.threshold]

#         rescored_transcription = []
#         for i in uncertain_tokens:
#             # Probability of token being correct from Whisper
#             whisper_prob = confidence[i]

#             token_text = whisper_prediction.split()[i]  # Get the text of the uncertain token
#             bert_prob = self.score_with_bert(token_text)  # Use BERT to predict the probability of the token

#             weighted_prob = (1 - self.gamma) * whisper_prob + self.gamma * bert_prob

#             rescored_transcription.append((i, weighted_prob))

#         return rescored_transcription

# def process_corpus_audio_files(data_dir):
#     rescorer = HybridRescorer()
#     for root, dirs, files in os.walk(data_dir):
#         for file in files:
#             if file.endswith(".wav"):
#                 audio_path = os.path.join(root, file)
#                 rescored_transcription = rescorer.hybrid_rescoring(audio_path)
#                 print(f"Audio File: {audio_path}")
#                 print(f"Rescored Transcription: {rescored_transcription}\n")

# if __name__ == "__main__":
#     data_dir = r"C:\Users\divya\Downloads\DS_10283_4836\edacc_v1.0\edacc_v1.0\data"
#     print("processing or trying?")
#     process_corpus_audio_files(data_dir)




#     # import os
# # import torch
# # import numpy as np
# # from transformers import BertTokenizer, BertForSequenceClassification, WhisperForConditionalGeneration, WhisperProcessor
# # import torch.nn.functional as F
# # import soundfile as sf
# # from scipy.signal import resample
# # from whisper import load_model
# # from transformers import WhisperTokenizer
# # from transformers import WhisperProcessor
# # import soundfile as sf
# # import torch
# # from transformers import WhisperForConditionalGeneration, WhisperProcessor
# # import librosa

# # class HybridRescorer:
# #     def __init__(self, bert_model_name='bert-base-uncased', whisper_model_type='openai/whisper-small', gamma=0.5):
# #         self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
# #         self.bert_model = BertForSequenceClassification.from_pretrained(bert_model_name)
# #         self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_type)
# #         self.processor = WhisperProcessor.from_pretrained(whisper_model_type)
# #         self.threshold = 0.5  # temporary Threshold rn for low confidence in Whisper prediction
# #         self.gamma = gamma  # Weight parameter for combining Whisper and BERT probabilities that have to fine tune


# #     def predict_with_whisper(self, audio_path):
# #     # Load and decode the audio file
# #         audio_input, sr = sf.read(audio_path)
# #         if sr != 16000:
# #              audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)

# #     # Process audio input
# #         inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt")
# #         input_features = inputs['input_features']

# #         decoder_input_ids = torch.full((input_features.size(0), 1), self.whisper_model.config.decoder_start_token_id, dtype=torch.long)

# #         with torch.no_grad():
# #         # Generate outputs using model.generate() to get token IDs
# #             generated_ids = self.whisper_model.generate(input_features, decoder_input_ids=decoder_input_ids)

# #         # Decode the generated ids to get the transcription text
# #             transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# #         # Obtain the logits from the model
# #             outputs = self.whisper_model(input_features=input_features, decoder_input_ids=decoder_input_ids)
# #             logits = outputs.logits

# #         # Compute the softmax probabilities for each token in the transcription
# #             probs = torch.nn.functional.softmax(logits, dim=-1)

# #         # Extract the maximum probability for each token
# #             max_prob_per_token = []
# #             for i in range(probs.size(1)):  # Iterate over tokens
# #              max_prob_per_token.append(torch.max(probs[:, i]).item())
# #         print(transcription)
# #         print(max_prob_per_token)
# #         return transcription, max_prob_per_token
# #     # def predict_with_whisper(self, audio_path):
# #     # # Load and decode the audio file
# #     #     audio_input, sr = sf.read(audio_path)
# #     #     if sr != 16000:
# #     #         audio_input = librosa.resample(audio_input, orig_sr=sr, target_sr=16000)
    
# #     # # Process audio input
# #     #     inputs = self.processor(audio_input, sampling_rate=16000, return_tensors="pt")
# #     #     input_features = inputs['input_features']

# #     #     decoder_input_ids = torch.full((input_features.size(0),1), self.whisper_model.config.decoder_start_token_id, dtype = torch.long)


# #     #     with torch.no_grad():
# #     #     # Generate outputs using model.generate() to get token IDs
# #     #         generated_ids = self.whisper_model.generate(input_features, decoder_input_ids=decoder_input_ids)

# #     #     # Decode the generated ids to get the transcription text
# #     #         transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# #     # # To get logits for probability calculation (if needed), you should use model() instead of model.generate()
# #     #     with torch.no_grad():
# #     #         outputs = self.whisper_model(input_features=input_features, decoder_input_ids=decoder_input_ids)
# #     #         logits = outputs.logits
# #     #         print(logits)
# #     #         probs = torch.nn.functional.softmax(logits, dim=-1)
# #     #         print(probs)
# #     #         max_prob_per_token = [prob.max().item() for prob in probs]  # Ensure this is a list
# #     #         print("AFFFFF",max_prob_per_token)
# #     #         print("fgnjsjfsfds", transcription)
# #     #     return transcription, max_prob_per_token        

    
# #     def score_with_bert(self, text):
# #         inputs = self.bert_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
# #         with torch.no_grad():
# #             outputs = self.bert_model(**inputs)
# #             probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
# #             return probabilities[:, 1].item()
    
# #     def hybrid_rescoring(self, audio_path):
# #         whisper_prediction, confidence = self.predict_with_whisper(audio_path)
# #         uncertain_tokens = [i for i, conf in enumerate(confidence) if conf < self.threshold]

# #         rescored_transcription = []
# #         for i in uncertain_tokens:
# #             # Probability of token being correct from Whisper
# #             whisper_prob = confidence[i]

# #             token_text = whisper_prediction.split()[i]  # Get the text of the uncertain token
# #             bert_prob = self.score_with_bert(token_text)  # Use BERT to predict the probability of the token

# #             weighted_prob = (1 - self.gamma) * whisper_prob + self.gamma * bert_prob

# #             rescored_transcription.append((i, weighted_prob))

# #         return rescored_transcription


# # def process_corpus_audio_files(data_dir):
# #     rescorer = HybridRescorer()
# #     for root, dirs, files in os.walk(data_dir):
# #         for file in files:
# #             if file.endswith(".wav"):
# #                 audio_path = os.path.join(root, file)
# #                 rescored_transcription = rescorer.hybrid_rescoring(audio_path)
# #                 print(f"Audio File: {audio_path}")
# #                 print(f"Rescored Transcription: {rescored_transcription}\n")

# # if __name__ == "__main__":
# #     data_dir = r"C:\Users\divya\Downloads\DS_10283_4836\edacc_v1.0\edacc_v1.0\data"
# #     print("processing or trying?")
# #     process_corpus_audio_files(data_dir)



