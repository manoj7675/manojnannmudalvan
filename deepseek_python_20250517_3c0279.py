import pandas as pd
import numpy as np
import random
from collections import defaultdict

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Emotion categories and their characteristics
emotions = {
    "happy": {
        "keywords": ["love", "great", "amazing", "wonderful", "happy", "excited", "yay", "awesome"],
        "emojis": ["😊", "😂", "😍", "❤️", "🎉", "👍", "✨"],
        "punctuation": ["!", "!!", "!!!", " :)", " :D"],
        "sentence_patterns": [
            "I'm so {keyword} about this!",
            "This is {a} {keyword} day!",
            "Feeling {keyword} right now {emoji}",
            "Can't believe how {keyword} this is {emoji}"
        ]
    },
    "sad": {
        "keywords": ["sad", "depressed", "lonely", "heartbroken", "cry", "miss", "hurt"],
        "emojis": ["😢", "😭", "💔", "😞", "🥺"],
        "punctuation": ["...", ".", " :(", " :'("],
        "sentence_patterns": [
            "Feeling so {keyword} today...",
            "I {keyword} when this happens",
            "Why am I so {keyword}? {emoji}",
            "Can't stop {keyword}ing about it"
        ]
    },
    "angry": {
        "keywords": ["angry", "mad", "hate", "furious", "annoyed", "upset", "disgusted"],
        "emojis": ["😠", "😡", "🤬", "👿", "💢"],
        "punctuation": ["!", "!!", "!!!", " >:("],
        "sentence_patterns": [
            "This makes me {keyword}!",
            "I'm so {keyword} right now {emoji}",
            "Can't believe how {keyword} this makes me",
            "{keyword} at this situation!"
        ]
    },
    "fearful": {
        "keywords": ["scared", "afraid", "anxious", "terrified", "worried", "nervous"],
        "emojis": ["😨", "😰", "😱", "😳", "🥶"],
        "punctuation": ["...", ".", " :/", " :S"],
        "sentence_patterns": [
            "I'm feeling {keyword} about this...",
            "This makes me {keyword} {emoji}",
            "So {keyword} right now",
            "Getting {keyword} just thinking about it"
        ]
    },
    "surprised": {
        "keywords": ["surprised", "shocked", "amazed", "astonished", "wow", "unbelievable"],
        "emojis": ["😲", "😮", "🤯", "😳", "👀"],
        "punctuation": ["!", "!!", "!!!", " :O"],
        "sentence_patterns": [
            "Wow, I'm {keyword}!",
            "This is {a} {keyword} result!",
            "Didn't see that coming {emoji}",
            "Totally {keyword} by this!"
        ]
    },
    "neutral": {
        "keywords": ["post", "update", "share", "information", "news", "status"],
        "emojis": ["📝", "📰", "ℹ️", "🔍"],
        "punctuation": [".", ",", ""],
        "sentence_patterns": [
            "Just {keyword}ing my thoughts",
            "Here's {a} {keyword} for today",
            "Sharing this {keyword}",
            "My {keyword} on this topic"
        ]
    }
}

# Helper functions
def get_article(word):
    return "an" if word[0].lower() in ['a','e','i','o','u'] else "a"

def generate_sentence(emotion):
    pattern = random.choice(emotions[emotion]["sentence_patterns"])
    keyword = random.choice(emotions[emotion]["keywords"])
    
    replacements = {
        "{keyword}": keyword,
        "{a}": get_article(keyword),
        "{emoji}": random.choice(emotions[emotion]["emojis"])
    }
    
    for k, v in replacements.items():
        pattern = pattern.replace(k, v)
    
    # Add punctuation
    pattern += random.choice(emotions[emotion]["punctuation"])
    
    # Randomly add hashtags or mentions
    if random.random() > 0.7:
        pattern += " #" + keyword.lower()
    if random.random() > 0.8:
        pattern += " @" + random.choice(["user123", "friend", "officialpage"])
    
    # Randomly make some letters uppercase for emphasis
    if random.random() > 0.8 and emotion != "neutral":
        words = pattern.split()
        for i in range(len(words)):
            if random.random() > 0.7:
                words[i] = words[i].upper()
        pattern = " ".join(words)
    
    return pattern

# Generate the dataset
def generate_dataset(num_samples=5000):
    data = []
    emotion_distribution = {
        "happy": 0.25,
        "sad": 0.2,
        "angry": 0.15,
        "fearful": 0.1,
        "surprised": 0.1,
        "neutral": 0.2
    }
    
    emotions_list = list(emotion_distribution.keys())
    probs = list(emotion_distribution.values())
    
    for _ in range(num_samples):
        emotion = np.random.choice(emotions_list, p=probs)
        text = generate_sentence(emotion)
        
        # Add some noise (random words from other emotions)
        if random.random() > 0.8 and emotion != "neutral":
            other_emotion = random.choice([e for e in emotions_list if e != emotion])
            noise_word = random.choice(emotions[other_emotion]["keywords"])
            text = f"{noise_word} {text}" if random.random() > 0.5 else f"{text} {noise_word}"
        
        data.append({
            "text": text,
            "emotion": emotion,
            "platform": random.choice(["Twitter", "Facebook", "Instagram", "Reddit", "TikTok"]),
            "time_of_day": random.choice(["morning", "afternoon", "evening", "night"]),
            "has_emoji": 1 if any(e in text for e in ["😊","😢","😠","😨","😲"]) else 0,
            "has_hashtag": 1 if "#" in text else 0,
            "has_mention": 1 if "@" in text else 0,
            "text_length": len(text),
            "num_exclamation": text.count("!"),
            "num_question": text.count("?"),
        })
    
    return pd.DataFrame(data)

# Generate and save dataset
print("Generating dataset...")
dataset = generate_dataset(5000)

# Save to CSV
dataset.to_csv("social_media_emotion_dataset.csv", index=False)
print("Dataset saved as 'social_media_emotion_dataset.csv'")

# Show sample data
print("\nSample dataset entries:")
print(dataset.head(10))

# Show emotion distribution
print("\nEmotion distribution:")
print(dataset["emotion"].value_counts())

# Basic visualization
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(data=dataset, x="emotion", order=dataset["emotion"].value_counts().index)
plt.title("Distribution of Emotions in Dataset")
plt.show()

# Platform distribution
plt.figure(figsize=(10, 6))
sns.countplot(data=dataset, x="platform", hue="emotion")
plt.title("Emotion Distribution by Platform")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()