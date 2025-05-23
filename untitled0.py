# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/149ceknb_NoPh3mQpIZa2-nASxzw3vJDF
"""

import time
import re
import torch
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from transformers import RobertaTokenizer, RobertaModel, AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap

# Load spaCy model for grammar and lexical pattern analysis
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model is not installed, download it
    import subprocess
    subprocess.run("python -m spacy download en_core_web_sm", shell=True)
    nlp = spacy.load("en_core_web_sm")

# Load PolitiTweet dataset
polititweet_df = pd.read_csv("/content/PAk_Tweets_local_data (1).csv")  # Replace with actual path

# Preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s.,!?;:'\"()]", "", text)  # Keep some punctuation for grammar analysis
    return text

# Apply preprocessing
polititweet_df["text"] = polititweet_df["text"].apply(preprocess_text)

# Handle missing values
polititweet_df = polititweet_df.dropna(subset=["category"])

# Map boolean labels to integers
polititweet_df["category"] = polititweet_df["category"].astype(int)

# Check class distribution
print("Class distribution in PolitiTweet dataset:")
print(polititweet_df["category"].value_counts())

# Enhanced linguistic and grammatical feature extraction
def extract_linguistic_features(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        # Return default values if text is empty or not a string
        return {
            'capitalization_ratio': 0,
            'exclamation_count': 0,
            'question_mark_count': 0,
            'clickbait_score': 0,
            'emotional_language': 0,
            'hyperbolic_score': 0,
            'hedging_words': 0,
            'certainty_words': 0,
            'partisan_score': 0,
            'complexity_score': 0,
            'avg_sentence_length': 0,
            'pronoun_ratio': 0,
            'verb_ratio': 0,
            'adjective_ratio': 0,
            'adverb_ratio': 0
        }

    features = {}

    # Capitalization and punctuation analysis
    words = text.split()
    features['capitalization_ratio'] = sum(1 for word in words if word.isupper()) / max(len(words), 1)
    features['exclamation_count'] = text.count('!')
    features['question_mark_count'] = text.count('?')

    # Presence of clickbait phrases
    clickbait_phrases = ["you won't believe", "shocking", "mind blowing", "this is why",
                         "must see", "breaking", "exclusive", "secret", "revealed",
                         "urgent", "alert", "warning", "attention", "outrageous"]
    features['clickbait_score'] = sum(1 for phrase in clickbait_phrases if phrase in text.lower())

    # Emotional language markers
    emotional_words = ["angry", "sad", "happy", "excited", "outraged", "furious",
                       "terrible", "horrible", "amazing", "incredible", "unbelievable",
                       "disgusting", "devastating", "thrilled", "ecstatic", "heartbreaking"]
    features['emotional_language'] = sum(1 for word in emotional_words if word in text.lower().split())

    # Hyperbolic expressions
    hyperbolic_phrases = ["best ever", "worst ever", "never before", "absolutely",
                          "completely", "totally", "utterly", "unparalleled", "unprecedented",
                          "extraordinary", "extreme", "perfect", "enormous", "gigantic", "massive"]
    features['hyperbolic_score'] = sum(1 for phrase in hyperbolic_phrases if phrase in text.lower())

    # Hedging words (often used in uncertain claims)
    hedging_words = ["may", "might", "could", "possibly", "perhaps", "allegedly",
                     "reportedly", "seems", "appears", "likely", "unlikely", "supposedly"]
    features['hedging_words'] = sum(1 for word in hedging_words if word in text.lower().split())

    # Certainty expressions (often overused in misleading content)
    certainty_words = ["definitely", "certainly", "undoubtedly", "absolutely", "guaranteed",
                       "proven", "without a doubt", "unquestionably", "must", "always", "never"]
    features['certainty_words'] = sum(1 for word in certainty_words if word in text.lower().split())

    # Partisan/divisive language
    partisan_words = ["liberal", "conservative", "leftist", "right-wing", "radical",
                      "socialist", "communist", "fascist", "extremist", "conspiracy",
                      "deep state", "mainstream media", "corruption", "elite", "patriot", "traitor"]
    features['partisan_score'] = sum(1 for word in partisan_words if word in text.lower().split())

    # Run spaCy for more sophisticated grammatical analysis
    try:
        doc = nlp(text)

        # Text complexity measures
        features['complexity_score'] = sum(len(token.text) for token in doc) / max(len(doc), 1)

        # Sentence structure analysis
        sentences = list(doc.sents)
        features['avg_sentence_length'] = sum(len(sentence) for sentence in sentences) / max(len(sentences), 1) if sentences else 0

        # Parts of speech ratios
        pos_counts = {}
        for token in doc:
            pos = token.pos_
            pos_counts[pos] = pos_counts.get(pos, 0) + 1

        total_tokens = len(doc)
        features['pronoun_ratio'] = pos_counts.get('PRON', 0) / max(total_tokens, 1)
        features['verb_ratio'] = pos_counts.get('VERB', 0) / max(total_tokens, 1)
        features['adjective_ratio'] = pos_counts.get('ADJ', 0) / max(total_tokens, 1)
        features['adverb_ratio'] = pos_counts.get('ADV', 0) / max(total_tokens, 1)

    except Exception as e:
        print(f"Error in spaCy processing: {e}")
        # Fallback values
        features['complexity_score'] = 0
        features['avg_sentence_length'] = 0
        features['pronoun_ratio'] = 0
        features['verb_ratio'] = 0
        features['adjective_ratio'] = 0
        features['adverb_ratio'] = 0

    return features

# Add linguistic features to the dataset
print("Extracting linguistic features...")
linguistic_features = polititweet_df["text"].apply(extract_linguistic_features)
polititweet_df = pd.concat([polititweet_df, pd.DataFrame(linguistic_features.tolist())], axis=1)

# Split data into train and test sets
train_df, test_df = train_test_split(polititweet_df, test_size=0.2, random_state=42, stratify=polititweet_df["category"])

# Load RoBERTa tokenizer
print("Loading RoBERTa tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Tokenize data
def tokenize_data(df, tokenizer, max_length=128):
    return tokenizer(
        df["text"].tolist(),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

train_encodings = tokenize_data(train_df, tokenizer)
test_encodings = tokenize_data(test_df, tokenizer)

# Get all linguistic feature columns
linguistic_feature_columns = [col for col in train_df.columns if col not in ["text", "category"]]

# Combine RoBERTa embeddings and linguistic features
def combine_features(encodings, df, feature_columns):
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    linguistic_features = torch.tensor(df[feature_columns].values, dtype=torch.float32)
    return input_ids, attention_mask, linguistic_features

print("Combining features...")
train_input_ids, train_attention_mask, train_linguistic = combine_features(train_encodings, train_df, linguistic_feature_columns)
test_input_ids, test_attention_mask, test_linguistic = combine_features(test_encodings, test_df, linguistic_feature_columns)

# Create DataLoader
def create_dataloader(input_ids, attention_mask, linguistic_features, labels, batch_size=16, shuffle=True):
    dataset = TensorDataset(input_ids, attention_mask, linguistic_features, torch.tensor(labels, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

train_loader = create_dataloader(train_input_ids, train_attention_mask, train_linguistic, train_df["category"].values)
test_loader = create_dataloader(test_input_ids, test_attention_mask, test_linguistic, test_df["category"].values)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Compute class weights
unique_classes = np.unique(train_df["category"])
class_weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=train_df["category"])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Enhanced RoBERTa Model with linguistic features
class EnhancedRoBERTa(torch.nn.Module):
    def __init__(self, roberta_model, linguistic_features_dim, hidden_dim=256, num_classes=2):
        super(EnhancedRoBERTa, self).__init__()
        self.roberta = roberta_model
        self.linear1 = torch.nn.Linear(768 + linguistic_features_dim, hidden_dim)  # 768 (RoBERTa) + linguistic features
        self.dropout = torch.nn.Dropout(0.3)
        self.linear2 = torch.nn.Linear(hidden_dim, num_classes)
        self.relu = torch.nn.ReLU()
        self.feature_names = None

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def forward(self, input_ids, attention_mask, linguistic_features):
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        pooled_output = roberta_output.last_hidden_state[:, 0, :]

        # Combine RoBERTa embeddings and linguistic features
        combined_features = torch.cat([pooled_output, linguistic_features], dim=1)

        x = self.linear1(combined_features)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.linear2(x)
        return logits, roberta_output.attentions

# Function to visualize attention weights
def visualize_attention_weights(text, input_ids, attentions, tokenizer, device, layer_idx=-1, head_idx=0):
    # Convert input IDs to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get attention weights for a specific layer and head
    attention = attentions[layer_idx][0, head_idx].detach().cpu()

    # Create attention heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention, xticklabels=tokens, yticklabels=tokens, cmap="YlGnBu")
    plt.title(f"Attention Weights (Layer {layer_idx+1}, Head {head_idx+1})")
    plt.tight_layout()
    plt.show()

# Load pre-trained RoBERTa model
# Load pre-trained RoBERTa model with explicit attention implementation
print("Loading RoBERTa model...")
roberta_model = RobertaModel.from_pretrained("roberta-base", attn_implementation="eager")

# Wrap the RoBERTa model in your custom EnhancedRoBERTa class
model = EnhancedRoBERTa(roberta_model, linguistic_features_dim=len(linguistic_feature_columns)).to(device)
# Set feature names for SHAP explainability
model.set_feature_names(linguistic_feature_columns)

# Optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training loop with validation and early stopping
num_epochs = 7
best_val_loss = float('inf')
patience = 3
counter = 0
train_losses = []
val_losses = []
val_accuracies = []

print("Starting training...")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
        input_ids, attention_mask, linguistic_features, labels = batch
        input_ids, attention_mask, linguistic_features, labels = input_ids.to(device), attention_mask.to(device), linguistic_features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = model(input_ids, attention_mask, linguistic_features)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    model.eval()
    val_loss = 0
    val_predictions, val_true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, linguistic_features, labels = batch
            input_ids, attention_mask, linguistic_features, labels = input_ids.to(device), attention_mask.to(device), linguistic_features.to(device), labels.to(device)

            outputs, _ = model(input_ids, attention_mask, linguistic_features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_predictions.extend(preds)
            val_true_labels.extend(labels.cpu().numpy())

    avg_train_loss = total_loss / len(train_loader)
    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = accuracy_score(val_true_labels, val_predictions)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Load the best model
model.load_state_dict(torch.load("best_model.pth"))

# Plot training metrics
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.tight_layout()
plt.show()

# Evaluate model on test set
model.eval()
y_true = []
y_pred = []
all_linguistic_features = []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, linguistic_features, labels = batch
        input_ids, attention_mask, linguistic_features, labels = input_ids.to(device), attention_mask.to(device), linguistic_features.to(device), labels.to(device)

        outputs, _ = model(input_ids, attention_mask, linguistic_features)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds)
        all_linguistic_features.extend(linguistic_features.cpu().numpy())

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# FIXED: SHAP explainer implementation
class FixedLinguisticFeatureExplainer:
    def __init__(self, model, feature_names, background_samples=None):
        self.model = model
        self.feature_names = feature_names
        self.background_samples = background_samples

    def get_shap_values(self, samples, device):
        # Create a background dataset (subset of all samples)
        if self.background_samples is None:
            background = samples[:100] if len(samples) > 100 else samples
        else:
            background = self.background_samples

        # Create a function that only takes the linguistic features
        def model_predict(X):
            batch_size = X.shape[0]
            dummy_input_ids = torch.ones((batch_size, 10), dtype=torch.long).to(device)
            dummy_attention_mask = torch.ones((batch_size, 10), dtype=torch.long).to(device)
            linguistic_tensor = torch.tensor(X, dtype=torch.float32).to(device)

            self.model.eval()
            with torch.no_grad():
                outputs, _ = self.model(dummy_input_ids, dummy_attention_mask, linguistic_tensor)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                # Return probability of being real news (class 1)
                return probs[:, 1]

        # Convert samples to numpy arrays if they're not already
        if isinstance(samples, list):
            samples = np.array(samples)
        if isinstance(background, list):
            background = np.array(background)

        # Create the SHAP explainer
        explainer = shap.KernelExplainer(model_predict, background)

        # Calculate SHAP values
        shap_values = explainer.shap_values(samples, nsamples=100)

        return shap_values, explainer

    def plot_shap_summary(self, shap_values):
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, feature_names=self.feature_names)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.show()

        # Bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, feature_names=self.feature_names, plot_type="bar")
        plt.title("SHAP Feature Importance (Bar)")
        plt.tight_layout()
        plt.show()

# Simple prediction function without visualizations
def predict_news(statement, model, tokenizer, device, feature_columns):
    # Preprocess text
    processed_statement = preprocess_text(statement)

    # Extract linguistic features
    linguistic_features = extract_linguistic_features(processed_statement)
    linguistic_tensor = torch.tensor([list(linguistic_features.values())], dtype=torch.float32).to(device)

    # Tokenize input
    inputs = tokenizer(processed_statement, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs, _ = model(input_ids, attention_mask, linguistic_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        prediction = torch.argmax(outputs, dim=1).item()

    # Results
    label = "Real" if prediction == 1 else "Fake"
    confidence = probs[0][prediction]
    return label, confidence

# Create SHAP explainer safely (using a small subset of data)
print("Creating SHAP explainer...")
sample_size = min(50, len(all_linguistic_features))  # Use smaller sample to avoid memory issues
feature_explainer = FixedLinguisticFeatureExplainer(
    model,
    feature_names=linguistic_feature_columns,
    background_samples=np.array(all_linguistic_features[:sample_size])
)

# Function for detailed explanation of a prediction with word-level contribution
def explain_prediction(statement, model, tokenizer, device, feature_columns, feature_explainer=None):
    # Preprocess text
    processed_statement = preprocess_text(statement)

    # Extract linguistic features
    linguistic_features = extract_linguistic_features(processed_statement)
    feature_values = np.array([list(linguistic_features.values())])
    linguistic_tensor = torch.tensor(feature_values, dtype=torch.float32).to(device)

    # Tokenize input
    inputs = tokenizer(processed_statement, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs, attentions = model(input_ids, attention_mask, linguistic_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        prediction = torch.argmax(outputs, dim=1).item()

    # Results
    label = "Real" if prediction == 1 else "Fake"
    confidence = probs[0][prediction]

    print(f"\nPrediction: {label} News (Confidence: {confidence:.2%})")

    # Display top linguistic features
    print("\nLinguistic features in this text:")
    feature_dict = dict(zip(feature_columns, list(linguistic_features.values())))
    for feature, value in sorted(feature_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
        if value > 0:  # Only show non-zero features
            print(f"- {feature}: {value:.4f}")

    # Word-level contribution analysis
    print("\nWord-level contribution analysis:")

    # Get tokens from the input
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    words = [tokenizer.convert_tokens_to_string(t).strip() for t in tokens]

    # Extract attention weights from the last layer
    # For simplicity, we'll use the average attention across all heads in the last layer
    last_layer_attn = attentions[-1].mean(dim=1)  # Average over all attention heads
    word_importances = last_layer_attn[0, 0, :].cpu().numpy()  # Attention from [CLS] token to all tokens

    # Normalize word importances to percentages
    word_importances = word_importances / word_importances.sum() * 100

    # Print the words and their contribution scores
    word_contributions = []
    for i, (word, importance) in enumerate(zip(words, word_importances)):
        if word not in ['<s>', '</s>', '<pad>']:  # Skip special tokens
            word_contributions.append((word, importance))

    # Sort words by importance
    word_contributions.sort(key=lambda x: x[1], reverse=True)

    # Print top contributing words
    print("Top words contributing to classification:")
    for word, importance in word_contributions[:10]:  # Show top 10 words
        print(f"- '{word}': {importance:.2f}%")

    # Visualize word importance
    plt.figure(figsize=(12, 6))
    plt.bar([w for w, _ in word_contributions[:15]], [i for _, i in word_contributions[:15]])
    plt.title(f"Top 15 Words Contributing to '{label}' Classification")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Visualize attention heatmap
    try:
        # Get attention weights for visualization
        # For simplicity, we'll use the first head in the last layer
        attention = attentions[-1][0, 0].detach().cpu()

        # Filter out padding tokens
        valid_length = attention_mask[0].sum().item()
        valid_tokens = tokens[:valid_length]

        # Create attention heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention[:valid_length, :valid_length],
                   xticklabels=valid_tokens,
                   yticklabels=valid_tokens,
                   cmap="YlGnBu")
        plt.title(f"Word-to-Word Attention (Last Layer)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not visualize attention: {e}")

    # SHAP explanation for linguistic features
    if feature_explainer is not None:
        try:
            # Get SHAP values for linguistic features
            shap_values, _ = feature_explainer.get_shap_values(feature_values, device)

            # Plot SHAP values
            print("\nLinguistic feature contributions to prediction:")
            plt.figure(figsize=(10, 6))
            shap.force_plot(np.mean(shap_values), feature_values, feature_names=feature_columns, matplotlib=True)
            plt.show()
        except Exception as e:
            print(f"SHAP explanation error: {e}")

    # Construct a more detailed textual explanation
    print("\nDetailed explanation:")

    # Identify suspicious/credible patterns
    suspicious_patterns = []
    credible_patterns = []

    # Check linguistic features
    if linguistic_features['emotional_language'] > 1:
        suspicious_patterns.append("contains emotional language")
    if linguistic_features['hyperbolic_score'] > 1:
        suspicious_patterns.append("uses hyperbolic expressions")
    if linguistic_features['clickbait_score'] > 0:
        suspicious_patterns.append("contains clickbait phrases")
    if linguistic_features['exclamation_count'] > 2:
        suspicious_patterns.append("uses multiple exclamation marks")
    if linguistic_features['capitalization_ratio'] > 0.2:
        suspicious_patterns.append("has unusual capitalization")
    if linguistic_features['partisan_score'] > 1:
        suspicious_patterns.append("contains partisan language")

    if linguistic_features['hedging_words'] > 1:
        credible_patterns.append("uses hedging language (showing caution)")
    if linguistic_features['verb_ratio'] > 0.2:
        credible_patterns.append("has a balanced use of verbs")
    if linguistic_features['complexity_score'] > 5:
        credible_patterns.append("shows linguistic complexity")

    # Print pattern analysis
    if prediction == 0:  # Fake news
        print("This statement was classified as potentially fake because it:")
        for pattern in suspicious_patterns:
            print(f"- {pattern}")
        print("\nHighly influential words that contributed to this classification:")
        for word, importance in word_contributions[:5]:
            print(f"- '{word}': {importance:.2f}%")
    else:  # Real news
        print("This statement was classified as likely real because it:")
        for pattern in credible_patterns:
            print(f"- {pattern}")
        print("\nHighly influential words that contributed to this classification:")
        for word, importance in word_contributions[:5]:
            print(f"- '{word}': {importance:.2f}%")

    # Provide a confidence assessment
    if confidence > 0.95:
        confidence_text = "The model is very confident in this classification."
    elif confidence > 0.7:
        confidence_text = "The model is reasonably confident in this classification."
    else:
        confidence_text = "The model is not very confident in this classification. Consider reviewing the content manually."

    print(f"\n{confidence_text}")

    return label, confidence, linguistic_features, word_contributions

# Enhanced prediction interface
def run_prediction_interface():
    print("\n" + "="*50)
    print("Fake News Detection System")
    print("="*50)
    print("Enter a news statement to check if it's real or fake (type 'exit' to quit):")

    while True:
        statement = input("\nEnter text: ")

        if statement.lower() == 'exit':
            break

        try:
            # First do quick prediction
            label, confidence = predict_news(statement, model, tokenizer, device, linguistic_feature_columns)
            print(f"Initial assessment: {label} News (Confidence: {confidence:.2%})")

            # Ask if user wants detailed explanation
            detailed = input("Would you like a detailed explanation? (y/n): ").lower() == 'y'

            if detailed:
                # Full explanation with visuals
                label, confidence, _, word_contributions = explain_prediction(
                    statement,
                    model,
                    tokenizer,
                    device,
                    linguistic_feature_columns,
                    feature_explainer
                )

                # Highlight words in the original text
                print("\nText with highlighted words:")

                # Create a copy of the original text
                highlighted_text = statement.split()

                # Sort words by importance
                important_words = [word for word, _ in word_contributions[:10]]

                # Print the text with important words highlighted
                for i, word in enumerate(highlighted_text):
                    word_clean = re.sub(r'[^\w]', '', word.lower())
                    if any(important_word in word_clean for important_word in important_words):
                        print(f"[{word}]", end=" ")
                    else:
                        print(word, end=" ")
                print("\n")

                print(f"\nFinal verdict: The statement is classified as {label} News with {confidence:.2%} confidence")

        except Exception as e:
            print(f"Error processing input: {e}")
            traceback.print_exc()

        print("\n" + "-"*50)

# Run the prediction interface
if __name__ == "__main__":
    run_prediction_interface()