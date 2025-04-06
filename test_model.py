import pickle

# Load model and vectorizer
with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Dummy JD and Resume (combine both)
text = """
We are looking for a data science expert skilled in Python, scikit-learn, NLP, machine learning, deep learning, and Flask.
Candidate should have experience with RNNs, LSTMs, and working on real-world analytics or forecasting projects.
"""

# Predict
vec = vectorizer.transform([text])
prob = model.predict_proba(vec)[0][1]

print(f"Predicted match score: {round(prob * 100, 2)}%")
