import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os
from classifier.preprocess import clean_text

def train_model(data_path='./data/IMDB_Dataset.csv', model_dir='model'):
    
    df = pd.read_csv(data_path)
    df['review'] = df['review'].apply(clean_text) # text preprocessing

    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    os.makedirs(model_dir, exist_ok=True)
    with open(f'{model_dir}/tfidf.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(f'{model_dir}/model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("===== Training completed")


if __name__ == "__main__":
    train_model()
