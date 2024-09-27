# model_training.py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from data_preprocessing import preprocess_data
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

# Set random seed for reproducibility
seed = 1234
np.random.seed(seed)

# Parameters for preprocessing
FILE_ID = "1pGn-1Rq2lC9AhsUtD5AcDEAb2Haxcyho"
OUTPUT = "airline_twitter_sentiment.csv"
COLUMN_NAME = "text"

# Preprocess data
X_train_original, X_val_original, X_test_original, y_train, y_val, y_test = preprocess_data(FILE_ID, OUTPUT, COLUMN_NAME)

# Define and train Multinomial Naive Bayes (Basic)
def train_multinomial_nb_basic(X_train, y_train, X_val, y_val):
    multi_nb = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)
    multi_nb.fit(X_train, y_train)
    y_pred_nb = multi_nb.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_nb)
    print("Multinomial Naive Bayes (Basic) Accuracy:", accuracy)
    return multi_nb

# Define and train Multinomial Naive Bayes (Advanced)
def train_multinomial_nb_advanced(X_train, y_train):
    # Define the number of splits and repetitions for repeated stratified k-fold cross-validation
    n_splits = 3  # Number of folds
    n_repeats = 1  # Number of repetitions

    # Initialize the repeated stratified k-fold cross-validator
    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    # Lists to store accuracy scores for each fold
    val_accuracy_scores = []

    # Iterate over the folds
    for train_index, val_index in rkf.split(X_train, y_train):
        # Get training and validation data for this fold
        X_train_fold = X_train[train_index].toarray()  # Convert sparse to dense
        X_val_fold = X_train[val_index].toarray()      # Convert sparse to dense
        y_train_fold = y_train.iloc[train_index]       
        y_val_fold = y_train.iloc[val_index]            

        # Create and fit the Multinomial Naive Bayes model
        model = MultinomialNB()
        model.fit(X_train_fold, y_train_fold)

        # Check results for Multinomial Naive Bayes
        val_pred = model.predict(X_val_fold)
        val_accuracy = accuracy_score(y_val_fold, val_pred)
        print(f'Accuracy on validation set (Multinomial Naive Bayes Advanced): {round(val_accuracy * 100, 4)}%')

        # Append accuracy score to list
        val_accuracy_scores.append(val_accuracy)

    # Return the mean validation accuracy
    return np.mean(val_accuracy_scores)


# Define and train Sklearn's GBM (Basic)
def train_gbm_basic(X_train, y_train, X_val, y_val):
    gbm = GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=seed)
    gbm.fit(X_train, y_train)
    y_pred = gbm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Gradient Boosting (Basic) Accuracy:", accuracy)
    return gbm

# Define and train Sklearn's GBM (Advanced)
def train_gbm_advanced(X_train, y_train):
    n_splits = 3  # Number of folds
    n_repeats = 1  # Number of repetitions

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    val_accuracy_scores = []

    for train_index, val_index in rkf.split(X_train, y_train):
        # Use direct indexing for the sparse matrix
        X_train_fold = X_train[train_index].toarray()  # Convert sparse to dense
        X_val_fold = X_train[val_index].toarray()      # Convert sparse to dense
        y_train_fold = y_train.iloc[train_index]       
        y_val_fold = y_train.iloc[val_index]            

        # Create and fit the Gradient Boosting model
        model = GradientBoostingClassifier()
        model.fit(X_train_fold, y_train_fold)

        # Check results for Gradient Boosting
        val_pred = model.predict(X_val_fold)
        val_accuracy = accuracy_score(y_val_fold, val_pred)
        print(f'Accuracy on validation set (Gradient Boosting Advanced): {round(val_accuracy * 100, 4)}%')

        # Append accuracy score to list
        val_accuracy_scores.append(val_accuracy)

    return np.mean(val_accuracy_scores)


# Define and train XGBoost (Basic)
def train_xgboost_basic(X_train, y_train, X_val, y_val):
    xgb_params = {
        'objective': 'multi:softmax',
        'eval_metric': 'mlogloss',
        'eta': 0.1,
        'max_depth': 6,
        'num_class': 3,
        'lambda': 0.8,
        'seed': seed
    }

    target_train = y_train.astype('category').cat.codes
    target_val = y_val.astype('category').cat.codes

    d_train = xgb.DMatrix(X_train, label=target_train)
    d_val = xgb.DMatrix(X_val, label=target_val)

    watchlist = [(d_train, 'train'), (d_val, 'validation')]
    bst = xgb.train(params=xgb_params,
                    dtrain=d_train,
                    num_boost_round=200,
                    evals=watchlist,
                    early_stopping_rounds=50,
                    verbose_eval=0)
    
    return bst

# Define and train XGBoost (Advanced)
def train_xgboost_advanced(X_train, y_train, seed=42):
    n_splits = 3  # Number of folds
    n_repeats = 1  # Number of repetitions

    param_grid = {
        'max_depth': [9],
        'eta': [0.1],
        'objective': ['multi:softmax'],
        'eval_metric': ['mlogloss'],
        'num_class': [3],
        'lambda': [0.1]
    }

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    mean_val_accuracies = []

    for params in ParameterGrid(param_grid):
        val_accuracy_scores = []
        fold_counter = 0

        for train_index, val_index in rkf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            fold_counter += 1

            print(f"Fold: {fold_counter}", ' | Hyperparameters:', params)

            target_train = y_train_fold.astype('category').cat.codes
            target_val = y_val_fold.astype('category').cat.codes

            d_train = xgb.DMatrix(X_train_fold, label=target_train)
            d_val = xgb.DMatrix(X_val_fold, label=target_val)
            watchlist = [(d_train, 'train'), (d_val, 'validation')]

            bst = xgb.train(params=params,
                            dtrain=d_train,
                            num_boost_round=400,
                            evals=watchlist,
                            early_stopping_rounds=50,
                            verbose_eval=0)

            train_pred = bst.predict(d_train)
            val_pred = bst.predict(d_val)

            train_accuracy = accuracy_score(target_train, train_pred)
            val_accuracy = accuracy_score(target_val, val_pred)
            val_accuracy_scores.append(val_accuracy)

            print(f'Accuracy on training set (XGBoost): {round(accuracy_score(target_train, train_pred) * 100, 4)}%')
            print(f'Accuracy on validation set (XGBoost): {round(val_accuracy * 100, 4)}%')

        mean_val_accuracy = np.mean(val_accuracy_scores)
        mean_val_accuracies.append(mean_val_accuracy)
        print("Mean Validation Accuracy (XGBoost Advanced):", mean_val_accuracy)
        print("====================================================================================================")

    best_params_index = np.argmax(mean_val_accuracies)
    best_params = list(ParameterGrid(param_grid))[best_params_index]
    best_mean_val_accuracy = mean_val_accuracies[best_params_index]

    print("Best Hyperparameters (XGBoost Advanced):", best_params)
    print("Best Mean Validation Accuracy:", best_mean_val_accuracy)

    return best_params



# Define and train Simple Neural Network (Basic)
def simple_neural_network_basic(X_train, y_train, X_val, y_val):
    # Generator for batches
    def batch_generator(X, y, batch_size, shuffle):
        number_of_batches = X.shape[0] // batch_size
        counter = 0
        sample_index = np.arange(X.shape[0])
        if shuffle:
            np.random.shuffle(sample_index)
        while True:
            batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
            X_batch = X[batch_index, :].toarray()
            y_batch = y[batch_index]
            counter += 1
            yield X_batch, y_batch
            if counter == number_of_batches:
                if shuffle:
                    np.random.shuffle(sample_index)
                counter = 0

    # One-hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded_train = onehot_encoder.fit_transform(y_train.reshape(-1, 1))
    onehot_encoded_val = onehot_encoder.transform(y_val.reshape(-1, 1))

    # Neural network architecture
    initializer = keras.initializers.he_normal(seed=seed)
    model = Sequential()
    model.add(Dense(20, activation='elu', kernel_initializer=initializer, input_dim=X_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax', kernel_initializer=initializer))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002),
                  loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    epochs = 50
    batch_size = 32
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_dense, onehot_encoded_train)).shuffle(buffer_size=10000).batch(batch_size)
    hist = model.fit(train_dataset, epochs=epochs, validation_data=(X_val_dense, onehot_encoded_val),
                     steps_per_epoch=X_train.shape[0] // batch_size)

    return model


# Define and train Simple Neural Network (Advanced)
def simple_neural_network_advanced(df, seed=42, n_splits=3, n_repeats=1, epochs=50, batch_size=128):
    df = shuffle(df, random_state=seed)
    test_set = df[:1000]
    df_train_val = df[1000:]
    y_test = test_set['airline_sentiment']
    
    vectorizer = TfidfVectorizer()
    X_test_fold = vectorizer.fit_transform(test_set['cleaned_text'])

    rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    train_scores = []
    val_scores = []
    best_fold_history = {'accuracy': [], 'val_accuracy': []}

    for fold, (train_index, val_index) in enumerate(rkf.split(df_train_val['cleaned_text'], df_train_val['airline_sentiment'])):
        print(f"Fold {fold + 1}:")
        X_train, X_val = df_train_val['cleaned_text'].iloc[train_index], df_train_val['cleaned_text'].iloc[val_index]
        y_train, y_val = df_train_val['airline_sentiment'].iloc[train_index], df_train_val['airline_sentiment'].iloc[val_index]

        X_train = vectorizer.fit_transform(X_train)
        X_val = vectorizer.transform(X_val)

        onehot_encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = onehot_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
        y_val_encoded = onehot_encoder.transform(y_val.to_numpy().reshape(-1, 1))

        model = Sequential([
            Dense(20, activation='elu', kernel_initializer='he_normal', input_dim=X_train.shape[1]),
            Dropout(0.5),
            Dense(3, activation='softmax', kernel_initializer='he_normal')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)

        hist = model.fit(X_train.toarray(), y_train_encoded,
                         validation_data=(X_val.toarray(), y_val_encoded),
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=[early_stopping],
                         verbose=0)

        fold_history = {'accuracy': hist.history['accuracy'], 'val_accuracy': hist.history['val_accuracy']}
        best_fold_history['accuracy'].append(fold_history['accuracy'])
        best_fold_history['val_accuracy'].append(fold_history['val_accuracy'])

        for epoch in range(len(hist.history['accuracy'])):
            print(f"Epoch {epoch + 1} - Train Accuracy: {hist.history['accuracy'][epoch]}, Validation Accuracy: {hist.history['val_accuracy'][epoch]}")

        train_scores.append(hist.history['accuracy'][-1])
        val_scores.append(hist.history['val_accuracy'][-1])

    return best_fold_history, train_scores, val_scores

def always_predict_negative(y_train, y_val):
    # Always predict 'negative'
    negative_pred_train = ['negative' for _ in range(len(y_train))]
    negative_pred_val = ['negative' for _ in range(len(y_val))]
    
    train_accuracy = accuracy_score(y_train, negative_pred_train)
    val_accuracy = accuracy_score(y_val, negative_pred_val)
    
    print(f'Accuracy on training set (Always Predict Negative): {round(train_accuracy * 100, 4)}%')
    print(f'Accuracy on validation set (Always Predict Negative): {round(val_accuracy * 100, 4)}%')


os.makedirs('models', exist_ok=True)


# Train all models
multi_nb_basic_model = train_multinomial_nb_basic(X_train_original, y_train, X_val_original, y_val)
joblib.dump(multi_nb_basic_model, 'models/multinomial_nb_basic_model.pkl')
multi_nb_advanced_params = train_multinomial_nb_advanced(X_train_original, y_train)
joblib.dump(train_multinomial_nb_advanced, 'models/train_multinomial_nb_advanced.pkl')

gbm_basic_model = train_gbm_basic(X_train_original, y_train, X_val_original, y_val)
joblib.dump(train_gbm_basic, 'models/train_gbm_basi.pkl')
gbm_advanced_params = train_gbm_advanced(X_train_original, y_train)
joblib.dump(train_gbm_advanced, 'models/train_gbm_advanced.pkl')

xgb_basic_model = train_xgboost_basic(X_train_original, y_train, X_val_original, y_val)
joblib.dump(train_xgboost_basic, 'models/train_xgboost_basic.pkl')
xgb_advanced_params = train_xgboost_advanced(X_train_original, y_train)


print("All models have been trained and saved.")



import sys

def main(model_name):
    if model_name == 'multi_nb_basic':
        train_multinomial_nb_basic(X_train_original, y_train, X_val_original, y_val)
    elif model_name == 'multi_nb_advanced':
        train_multinomial_nb_advanced(X_train_original, y_train)
    elif model_name == 'gbm_basic':
        train_gbm_basic(X_train_original, y_train, X_val_original, y_val)
    elif model_name == 'gbm_advanced':
        train_gbm_advanced(X_train_original, y_train)
    elif model_name == 'xgb_basic':
        train_xgboost_basic(X_train_original, y_train, X_val_original, y_val)
    elif model_name == 'xgb_advanced':
        train_xgboost_advanced(X_train_original, y_train)
    else:
        print("Invalid model name provided.")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python model_training.py <model_name>")
    else:
        main(sys.argv[1])

