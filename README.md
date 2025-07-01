Built a multi-label classification model to detect harmful content in social media comments using the Jigsaw dataset (~160,000 samples):

* Preprocessed text and fine-tuned bert-base-uncased with a custom classification head
* Predicted six toxicity categories: toxic, obscene, threat, insult, severe_toxic, identity_hate
* Tools: PyTorch, HuggingFace Transformers, Scikit-learn, Matplotlib
