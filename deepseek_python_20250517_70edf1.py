import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter

# Text processing and ML libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLP libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# Download NLTK resources
nltk.download(['punkt', 'stopwords', 'wordnet', 'vader_lexicon'])

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

## 1. Data Preparation (Example with synthetic social media data)
def generate_social_media_data(num_samples=5000):
    # Base emotions and sample phrases
    emotions = ['happy', 'sad', 'angry', 'fearful', 'surprised', 'neutral']
    
    happy_phrases = [
        "I'm so excited about this!", "What a wonderful day!", 
        "This makes me so happy :)", "Love this!", "Amazing news!"
    ]
    
    sad_phrases = [
        "Feeling really down today", "This is so depressing", 
        "I can't stop crying", "Why does this always happen to me?", "So heartbroken"
    ]
    
    angry_phrases = [
        "This makes me furious!", "I can't believe this nonsense", 
        "So angry right now", "What a terrible decision", "I hate this"
    ]
    
    fearful_phrases = [
        "This is really scary", "I'm terrified about what might happen", 
        "Anxious about the future", "This situation frightens me", "So worried"
    ]
    
    surprised_phrases = [
        "Wow, didn't see that coming!", "This is unexpected!", 
        "OMG what a surprise!", "I'm shocked!", "Didn't expect that at all"
    ]
    
    neutral_phrases = [
        "Just posting an update", "Here's a photo from today", 
        "Sharing this article", "Current status", "Regular day"
    ]
    
    # Create dataset
    data = []
    for _ in range(num_samples):
        emotion = np.random.choice(emotions, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.3])
        
        if emotion == 'happy':
            text = np.random.choice(happy_phrases)
        elif emotion == 'sad':
            text = np.random.choice(sad_phrases)
        elif emotion == 'angry':
            text = np.random.choice(angry_phrases)
        elif emotion == 'fearful':
            text = np.random.choice(fearful_phrases)
        elif emotion == 'surprised':
            text = np.random.choice(surprised_phrases)
        else:
            text = np.random.choice(neutral_phrases)
        
        # Add some noise and variations
        text = text.lower()
        if np.random.random() > 0.5:
            text += " " + " ".join(["lol", "omg", "smh", "wtf"][np.random.randint(0, 4)])
        if np.random.random() > 0.7:
            text = text.replace("!", "!!!")
        
        data.append({'text': text, 'emotion': emotion})
    
    return pd.DataFrame(data)

print("Generating synthetic social media data...")
df = generate_social_media_data(5000)

print("\nSample data:")
print(df.head())
print("\nEmotion distribution:")
print(df['emotion'].value_counts())

# Visualize emotion distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='emotion')
plt.title('Distribution of Emotions in Dataset')
plt.show()

## 2. Text Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

print("\nPreprocessing text data...")
df['cleaned_text'] = df['text'].apply(preprocess_text)

# Show before and after examples
print("\nText preprocessing examples:")
for i in range(3):
    print(f"Original: {df.iloc[i]['text']}")
    print(f"Cleaned: {df.iloc[i]['cleaned_text']}\n")

## 3. Feature Extraction
# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf.fit_transform(df['cleaned_text'])
y = df['emotion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

## 4. Traditional Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print("\nTraining traditional ML models...")
ml_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    ml_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'report': report
    }
    
    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=df['emotion'].unique(), 
                yticklabels=df['emotion'].unique())
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

## 5. Deep Learning Approach (LSTM)
print("\nTraining LSTM model...")

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_text'])

# Sequence conversion
X_seq = tokenizer.texts_to_sequences(df['cleaned_text'])
X_pad = pad_sequences(X_seq, maxlen=100, padding='post', truncating='post')

# Label encoding
y_encoded = pd.get_dummies(df['emotion'])

# Split for LSTM
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_pad, y_encoded, test_size=0.2, stratify=y, random_state=42
)

# Model architecture
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3)

# Train model
history = model.fit(
    X_train_lstm, y_train_lstm,
    epochs=15,
    batch_size=64,
    validation_data=(X_test_lstm, y_test_lstm),
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
y_pred_lstm = model.predict(X_test_lstm)
y_pred_classes = np.argmax(y_pred_lstm, axis=1)
y_test_classes = np.argmax(y_test_lstm.values, axis=1)

# Get emotion labels
emotion_labels = y_encoded.columns.tolist()
y_pred_labels = [emotion_labels[i] for i in y_pred_classes]
y_test_labels = [emotion_labels[i] for i in y_test_classes]

# Calculate metrics
lstm_accuracy = accuracy_score(y_test_labels, y_pred_labels)
lstm_report = classification_report(y_test_labels, y_pred_labels)

print("\nLSTM Model Results:")
print(f"Accuracy: {lstm_accuracy:.4f}")
print("Classification Report:")
print(lstm_report)

# Plot training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Training History')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

## 6. VADER Sentiment Analysis (Rule-based)
print("\nRunning VADER Sentiment Analysis...")

sia = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.5:
        return 'happy'
    elif compound <= -0.5:
        return 'angry'
    elif -0.5 < compound < -0.1:
        return 'sad'
    elif -0.1 <= compound <= 0.1:
        return 'neutral'
    else:
        return 'surprised'

df['vader_prediction'] = df['text'].apply(vader_sentiment)

vader_accuracy = accuracy_score(df['emotion'], df['vader_prediction'])
vader_report = classification_report(df['emotion'], df['vader_prediction'])

print("\nVADER Results:")
print(f"Accuracy: {vader_accuracy:.4f}")
print("Classification Report:")
print(vader_report)

## 7. Emotion Detection from Emojis
print("\nAnalyzing emojis for emotion detection...")

# Common emojis and their associated emotions
emoji_emotion_map = {
    '😊': 'happy', '😂': 'happy', '😍': 'happy', '❤️': 'happy',
    '😢': 'sad', '😭': 'sad', '💔': 'sad',
    '😠': 'angry', '😡': 'angry', '🤬': 'angry',
    '😨': 'fearful', '😱': 'fearful', '😰': 'fearful',
    '😲': 'surprised', '😮': 'surprised', '🤯': 'surprised'
}

def detect_emotion_from_emojis(text):
    emojis = [c for c in text if c in emoji_emotion_map]
    if not emojis:
        return None
    
    # Count emotion frequencies from emojis
    emotion_counts = Counter([emoji_emotion_map[e] for e in emojis])
    return emotion_counts.most_common(1)[0][0]

df['emoji_emotion'] = df['text'].apply(detect_emotion_from_emojis)

# Compare with actual emotions where emojis were found
emoji_results = df[df['emoji_emotion'].notnull()]
if not emoji_results.empty:
    emoji_accuracy = accuracy_score(
        emoji_results['emotion'], 
        emoji_results['emoji_emotion']
    )
    print(f"\nEmoji-based emotion detection accuracy: {emoji_accuracy:.2f}")
    print("Sample emoji matches:")
    print(emoji_results[['text', 'emotion', 'emoji_emotion']].head())
else:
    print("No emojis found in the dataset")

## 8. Model Comparison and Selection
print("\nModel Comparison:")
comparison = pd.DataFrame({
    'Model': list(ml_results.keys()) + ['LSTM', 'VADER'],
    'Accuracy': [res['accuracy'] for res in ml_results.values()] + [lstm_accuracy, vader_accuracy]
}).sort_values('Accuracy', ascending=False)

print(comparison)

# Select best model
best_model_name = comparison.iloc[0]['Model']
if best_model_name in ml_results:
    best_model = ml_results[best_model_name]['model']
else:
    best_model = model  # LSTM

print(f"\nBest performing model: {best_model_name}")

## 9. Saving the Best Model
import joblib

if best_model_name != 'LSTM':
    # Save traditional ML model
    joblib.dump(best_model, 'best_emotion_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
    print("\nSaved best model and vectorizer to disk")
else:
    # Save LSTM model and tokenizer
    model.save('emotion_lstm_model.h5')
    with open('tokenizer.pkl', 'wb') as f:
        joblib.dump(tokenizer, f)
    print("\nSaved LSTM model and tokenizer to disk")

## 10. Example Predictions
print("\nExample Predictions:")

sample_texts = [
    "I'm absolutely thrilled with this amazing news!!! 😍",
    "This is the worst day ever, I'm so upset 😠",
    "Feeling anxious about tomorrow's presentation...",
    "Wow! Didn't expect that at all! 😮",
    "Just sharing my lunch photo"
]

for text in sample_texts:
    cleaned = preprocess_text(text)
    
    if best_model_name != 'LSTM':
        # Traditional ML prediction
        features = tfidf.transform([cleaned])
        pred = best_model.predict(features)[0]
        proba = best_model.predict_proba(features).max()
    else:
        # LSTM prediction
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=100)
        pred_proba = model.predict(pad)[0]
        pred_idx = np.argmax(pred_proba)
        pred = emotion_labels[pred_idx]
        proba = pred_proba[pred_idx]
    
    # Emoji analysis
    emoji_emotion = detect_emotion_from_emojis(text)
    
    print(f"\nText: {text}")
    print(f"Predicted emotion: {pred} (confidence: {proba:.2f})")
    if emoji_emotion:
        print(f"Emoji suggests: {emoji_emotion}")