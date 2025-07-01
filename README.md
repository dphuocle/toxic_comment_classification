
# Toxic Comment Classification using BERT

This project fine-tunes a pre-trained BERT model to detect various types of toxicity in user-generated text, based on the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

The task is framed as a **multi-label classification problem**, where each comment may exhibit one or more of the following labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

## 💡 Objectives

- Fine-tune `bert-base-uncased` (HuggingFace Transformers) on the full training dataset.
- Predict multiple toxicity types simultaneously using a custom classification head.
- Evaluate the model using macro/micro F1-score and classification report.
- (Optional) Visualize attention weights or use SHAP for interpretability.

## 🔧 Tech Stack

- **Language**: Python
- **Model**: BERT (via HuggingFace `transformers`)
- **Frameworks**: PyTorch, scikit-learn
- **Hardware**: CUDA (NVIDIA GPU), tested on Colab / Kaggle (Tesla P100)

## 📂 Dataset

The dataset is provided by Kaggle and contains:
- `train.csv` – 160k+ labeled comments
- `test.csv`, `test_labels.csv` – for evaluation
- `sample_submission.csv` – for Kaggle submission

## 🚀 Training

Training is done over 3 epochs using:
- `BCEWithLogitsLoss` for multi-label classification
- AdamW optimizer with learning rate scheduler
- DataLoader with tokenized inputs (max_len = 128, batch_size = 32)

Training time (P100 GPU): approx. **30 mins/epoch × 3 = 1.5 hours**

## 📈 Results

Final evaluation is done on the validation split using:
- `classification_report` from scikit-learn
- Per-class precision, recall, and F1-score

## 📌 Future Work

- Use DistilBERT for faster training
- Add early stopping and model checkpointing
- Deploy as a web app with Streamlit for live toxicity analysis

## 🧠 Author
