from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import re

def main():
    print(lr_imdb_predict(["This movie was fantastic! I really loved it.",
                    "The film was terrible and I hated every minute of it."]))
    
def load_imdb_dataset():
    dataset = load_dataset("imdb")
    test_df = pd.DataFrame(dataset['test'])
    train_df = pd.DataFrame(dataset['train'])
    return train_df, test_df

#removing punctuation and lowercasing the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def imdb_ml_training():
    test_df, train_df = load_imdb_dataset()
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['cleaned_text'])
    X_test = vectorizer.transform(test_df['cleaned_text'])
    Y_train = train_df['label']
    Y_test = test_df['label']

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    # Plot prediction confusion matrix
    cm = confusion_matrix(Y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')
    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print("Classification Report:\n", classification_report(Y_test, y_pred))
    return model, vectorizer

def save_model(model, vectorizer):
    model_path = './data/models/imdb_lr_sentiment_model.joblib'
    vectorizer_path = './data/vectorizers/imdb_tfidf_vectorizer.joblib'

    if(not model or not vectorizer):
        raise ValueError("Model or vectorizer is None, cannot save.")
    
    # Save only if the files do not already exist
    if(Path(model_path).is_file() == False):
        joblib.dump(model, model_path)
    if(Path(vectorizer_path).is_file() == False):
        joblib.dump(vectorizer, vectorizer_path)

def lr_imdb_predict(texts):
    # Load model and vectorizer
    model_path = './data/models/imdb_lr_sentiment_model.joblib'
    vectorizer_path = './data/vectorizers/imdb_tfidf_vectorizer.joblib'

    # If model or vectorizer do not exist, train and save them
    if(Path(model_path).is_file() == False or Path(vectorizer_path).is_file() == False):
        model, vectorizer = imdb_ml_training()
        save_model(model, vectorizer)

    # Load the model and vectorizer
    model = joblib.load(model_path)   
    vectorizer = joblib.load(vectorizer_path)

    # Preprocess and predict
    cleaned_texts = [clean_text(text) for text in texts]
    X_test = vectorizer.transform(cleaned_texts)
    prediction = model.predict(X_test)
    return prediction

if __name__ == "__main__":
    main()
