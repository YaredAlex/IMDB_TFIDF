import pickle
from .preprocess import clean_text

def load_model(model_dir='model'):
    with open(f'{model_dir}/tfidf.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open(f'{model_dir}/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

def predict_sentiment(text, model_dir='model'):
    vectorizer, model = load_model(model_dir)
    cleaned = clean_text(text)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]
    confidence = max(model.predict_proba(transformed)[0])
    return prediction, confidence
