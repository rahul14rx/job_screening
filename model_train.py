import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load the dataset
df = pd.read_csv('dataset/resume_dataset.csv')  # Adjust path if needed
df = df[df['Category'].notnull()]  # Drop empty labels

# Binary classification: Data Science = 1, Others = 0
df['label'] = df['Category'].apply(lambda x: 1 if x.lower() in [
    'data science', 'machine learning', 'deep learning', 'nlp', 'blockchain', 'analytics'
] else 0)

# Text processing
X = df['Resume']
y = df['label']

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)

X_vec = vectorizer.fit_transform(X)

# Split & Train
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
print("Model Performance:")
print(classification_report(y_test, model.predict(X_test)))

# Save model and vectorizer
with open('model/model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model & vectorizer saved.")
