#Fake News Detection on Political Tweets
This project focuses on detecting fake news within political tweets using a hybrid approach combining a fine-tuned RoBERTa model and handcrafted linguistic features. The aim is to improve classification accuracy by enriching transformer-based embeddings with text-based cues.

📂 Dataset
Dataset Name: Political Tweets Dataset
Description:
The dataset contains political tweets labeled as either fake or real. Each entry includes the tweet text. This makes it suitable for training and evaluating machine learning models in fake news detection tasks.

🚀 Features Used

🔤 Linguistic Features
Part-of-Speech (POS) tag distributions
Named Entity Recognition (NER) counts
Lexical richness (type-token ratio)
Average word length and sentence length
Count of punctuation, uppercase words, hashtags, and mentions
Readability scores (Flesch Reading Ease)

🤖 RoBERTa Features
Pre-trained roberta-base embeddings
Fine-tuned on political tweet dataset
CLS token representation for downstream classification

⚙️ Methodology
Preprocessing
Tokenization using RoBERTa tokenizer
SpaCy used for linguistic annotations
Lowercasing, stopword removal, and filtering symbols
Feature Extraction
Extracted handcrafted linguistic features
Combined with transformer-based embeddings
Modeling
Multi-layer neural network with concatenated feature inputs
Fine-tuned roberta-base using Hugging Face Transformers
Used dropout, batch normalization, and early stopping
Evaluation
Split into training and test sets
Evaluated using accuracy, precision, recall, and F1-score

📈 Results
Metric	Value (example)
Accuracy	98.2%
Precision	97%
Recall	98%

🛠️ Tech Stack
Python
PyTorch
Hugging Face Transformers
Scikit-learn
SpaCy
Numpy & Pandas
Matplotlib / Seaborn
