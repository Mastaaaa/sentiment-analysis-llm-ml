from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import re

def main():
    test_df, train_df = load_imdb_dataset()
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    test_df['cleaned_text'] = test_df['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['cleaned_text'])
    X_test = vectorizer.transform(test_df['cleaned_text'])

    Y_train = train_df['label']
    Y_test = test_df['label']

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)


    cm = confusion_matrix(Y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues')
    disp.savefig("grafico.png")  

    print("Accuracy:", accuracy_score(Y_test, y_pred))
    print("Classification Report:\n", classification_report(Y_test, y_pred))



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


if __name__ == "__main__":
    main()
