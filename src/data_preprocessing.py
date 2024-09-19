# For notebook plotting
%matplotlib inline

# Standard packages
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# nltk for preprocessing of text data
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# sklearn for preprocessing and machine learning models
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics  import accuracy_score
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# XGBoost for Machine Learning (Gradient Boosting Machine (GBM))
import xgboost as xgb

# Keras for neural networks
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Flatten
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Random seeds for consistent results
#from tensorflow import set_random_seed
seed = 1234
np.random.seed(seed)
tf.random.set_seed(seed)



#Data Exploration 

"""For this section we are mostly interested in feature selection and the 
distribution of the target variable (airline_sentiment). If the target 
variable distribution is realistic and plausible we will move on the text 
preprocessing."""

import gdown

# Google Drive file ID
file_id = "1pGn-1Rq2lC9AhsUtD5AcDEAb2Haxcyho"
download_url = f"https://drive.google.com/uc?id={file_id}&export=download"

# Download the file using gdown
output = 'airline_twitter_sentiment.csv'
gdown.download(download_url, output, quiet=False)

# Load the CSV file into a DataFrame
try:
    raw_df = pd.read_csv(output)

    # Display the first few rows of the DataFrame
    print(raw_df.head())
except Exception as e:
    print("Error:", e)
    

# Select features
df = raw_df[['tweet_id', 'text', 'airline_sentiment']]
print('Feature selected DataFrame:')
df.head(2)


#Preprocessing

"""We created a preprocessor class to perform all steps that need to be 
performed before the text data can be vectorized. These preprocessing steps 
include:
-Tokenization
-Removing stop words
-Stemming
-Transform the tokens back to one string"""


class PreProcessor:
    '''
    Easily performs all the standard preprocessing steps
    like removing stopwords, stemming, etc.
    Only input that you need to provide is the dataframe and column name for the tweets
    '''
    def __init__(self, df, column_name):
        self.data = df
        self.conversations = list(self.data[column_name])
        self.stopwords = set(stopwords.words('english'))
        self.stemmer = SnowballStemmer("english")
        self.preprocessed = []

    def tokenize(self, sentence):
        '''
        Splits up words and makes a list of all words in the tweet
        '''
        tokenized_sentence = word_tokenize(sentence)
        return tokenized_sentence

    def remove_stopwords(self, sentence):
        '''Removes stopwords like 'a', 'the', 'and', etc.'''
        filtered_sentence = []
        for w in sentence:
            if w not in self.stopwords and len(w) > 1 and w[:2] != '//' and w != 'https':
                filtered_sentence.append(w)
        return filtered_sentence

    def stem(self, sentence):
        '''
        Stems certain words to their root form.
        For example, words like 'computer', 'computation'
        all get truncated to 'comput'
        '''
        return [self.stemmer.stem(word) for word in sentence]

    def join_to_string(self, sentence):
        '''
        Joins the tokenized words to one string.
        '''
        return ' '.join(sentence)

    def full_preprocess(self, n_rows=None):
        '''
        Preprocess a selected number of rows and
        connects them back to strings
        '''
        # If nothing is given do it for the whole dataset
        if n_rows == None:
            n_rows = len(self.data)

        # Perform preprocessing
        for i in range(n_rows):
            tweet = self.conversations[i]
            tokenized = self.tokenize(tweet)
            cleaned = self.remove_stopwords(tokenized)
            stemmed = self.stem(cleaned)
            joined = self.join_to_string(stemmed)
            self.preprocessed.append(joined)
        return self.preprocessed


# Instantiate PreProcessor with your DataFrame and column name
processor = PreProcessor(df, 'text')

# Perform full preprocessing
preprocessed_tweets = processor.full_preprocess()


# Preprocess text and put it in a new column
preprocessor = PreProcessor(df, 'text')
#df['cleaned_text'] = preprocessor.full_preprocess()
# Use .loc to set values instead of chained indexing
df.loc[:, 'cleaned_text'] = preprocessor.full_preprocess()



#Split data into train, validation and test sets

"""We split the data into a training set, validation set and test set. 
This is crucial for training and evaluation of good machine learning models.
The data will be shuffled and we take 1000 tweets for our test set. The 
remaining data will be split 80/20. This means that 80% will be used for 
training and 20% for validation set. During the model training phase our 
goal is to maximize the accuracy on the validation set. The test set is to 
check that our model truly generalizes to data it has never seen before."""



# Shuffling so we can get random tweets for the test set
df = shuffle(df, random_state=seed)
# Keep 1000 samples of the data as test set
test_set = df[:1000]


# Get training and validation data
X_train, X_val, y_train, y_val = train_test_split(df['cleaned_text'][1000:],
                                                  df['airline_sentiment'][1000:],
                                                  test_size=0.2,
                                                  random_state=seed)

# Get sentiment labels for test set
y_test = test_set['airline_sentiment']



#Data vectorization


# Create matrix based on word frequency in tweets
vectorizer = TfidfVectorizer()
X_train_original = vectorizer.fit_transform(X_train)
X_val_original = vectorizer.transform(X_val)
X_test_original = vectorizer.transform(test_set['cleaned_text'])

# Print the size of our data
print(f'Training size: {X_train_original.shape[0]} tweets\n\
Validation size: {X_val_original.shape[0]} tweets\n\
Test size: {X_test_original.shape[0]} tweets\n\
Amount of words (columns): {X_train_original.shape[1]} words')