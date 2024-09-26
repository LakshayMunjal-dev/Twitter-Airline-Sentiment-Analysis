# data_preprocessing.py

import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk
import gdown

nltk.download('stopwords')
nltk.download('punkt')

def download_data(file_id, output):
    """Download data from Google Drive."""
    download_url = f"https://drive.google.com/uc?id={file_id}&export=download"
    gdown.download(download_url, output, quiet=False)

def load_data(file_path):
    """Load data from a CSV file into a DataFrame."""
    try:
        raw_df = pd.read_csv(file_path)
        return raw_df
    except Exception as e:
        raise ValueError(f"Error loading data: {e}")

class PreProcessor:
    def __init__(self, df, column_name):
        self.data = df
        self.conversations = list(self.data[column_name])
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        self.preprocessed = []

    def tokenize(self, sentence):
        """Tokenize sentences into words."""
        return word_tokenize(sentence)

    def remove_stopwords(self, sentence):
        """Remove stopwords from the tokenized sentence."""
        return [w for w in sentence if w not in self.stopwords and len(w) > 1 and w[:2] != '//' and w != 'https']

    def stem(self, sentence):
        """Stem words to their root form."""
        return [self.stemmer.stem(word) for word in sentence]

    def join_to_string(self, sentence):
        """Join tokens into a single string."""
        return ' '.join(sentence)

    def full_preprocess(self, n_rows=None):
        """Perform full preprocessing on the dataset."""
        if n_rows is None:
            n_rows = len(self.data)
        for i in range(n_rows):
            tweet = self.conversations[i]
            tokenized = self.tokenize(tweet)
            cleaned = self.remove_stopwords(tokenized)
            stemmed = self.stem(cleaned)
            joined = self.join_to_string(stemmed)
            self.preprocessed.append(joined)
        return self.preprocessed

def preprocess_data(file_id, output, column_name):
    """Preprocess data, including downloading, loading, and transforming."""
    download_data(file_id, output)
    raw_df = load_data(output)
    df = raw_df[['tweet_id', 'text', 'airline_sentiment']]
    
    processor = PreProcessor(df, column_name)
    df['cleaned_text'] = processor.full_preprocess()
    
    df = shuffle(df, random_state=1234)
    test_set = df[:1000]

    X_train, X_val, y_train, y_val = train_test_split(df['cleaned_text'][1000:], df['airline_sentiment'][1000:], test_size=0.2, random_state=1234)
    y_test = test_set['airline_sentiment']

    vectorizer = TfidfVectorizer()
    X_train_original = vectorizer.fit_transform(X_train)
    X_val_original = vectorizer.transform(X_val)
    X_test_original = vectorizer.transform(test_set['cleaned_text'])

    return X_train_original, X_val_original, X_test_original, y_train, y_val, y_test
