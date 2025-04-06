from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_scores(jd_keywords, resume_keywords):
    corpus = [' '.join(jd_keywords), ' '.join(resume_keywords)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]
