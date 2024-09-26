# model_training.py

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ParameterGrid, RepeatedStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib  # For saving models
from data_preprocessing import preprocess_data  # Import preprocessing function

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
        y_train_fold = y_train.iloc[train_index]       # Assuming y_train is a DataFrame/Series
        y_val_fold = y_train.iloc[val_index]            # Assuming y_train is a DataFrame/Series

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
        y_train_fold = y_train.iloc[train_index]       # Assuming y_train is a DataFrame/Series
        y_val_fold = y_train.iloc[val_index]            # Assuming y_train is a DataFrame/Series

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
def train_xgboost_advanced(X_train, y_train):
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


# Train all models
multi_nb_basic_model = train_multinomial_nb_basic(X_train_original, y_train, X_val_original, y_val)
multi_nb_advanced_params = train_multinomial_nb_advanced(X_train_original, y_train)

gbm_basic_model = train_gbm_basic(X_train_original, y_train, X_val_original, y_val)
gbm_advanced_params = train_gbm_advanced(X_train_original, y_train)

xgb_basic_model = train_xgboost_basic(X_train_original, y_train, X_val_original, y_val)
xgb_advanced_params = train_xgboost_advanced(X_train_original, y_train)

# Save models using joblib
import joblib

joblib.dump(multi_nb_basic_model, 'models/multinomial_nb_basic_model.pkl')
joblib.dump(gbm_basic_model, 'models/gbm_basic_model.pkl')
joblib.dump(xgb_basic_model, 'models/xgb_basic_model.pkl')

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


"""#Multinomial Naive Bayes

# Multinomial Naive Bayes
multi_nb = MultinomialNB(alpha=1, class_prior=None, fit_prior=True)
multi_nb.fit(X_train_original, y_train)


#Multinomial Naive Bayes -- Updated

from sklearn.model_selection import RepeatedStratifiedKFold,ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Define the number of splits and repetitions for repeated stratified k-fold cross-validation
n_splits = 30  # Number of folds
n_repeats = 1  # Number of repetitions
param_grid = {
    # 'alphas': [0.75, 0.85, 1.0]  # Alpha values to try
    'alphas': [0.45, 0.58, 0.75, 0.85, 0.93, 1]
}
fold_counter = 0
repeat_counter = 0
# Initialize the repeated stratified k-fold cross-validator
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

# Lists to store accuracy scores for each fold
mean_val_accuracies = []
mean_train_accuracies = []


for params in ParameterGrid(param_grid):
  val_accuracy_scores = []
  train_accuracy_scores = []
  # Iterate over the folds
  for train_index, val_index in rkf.split(X_train, y_train):
      # Get training and validation data for this fold
      X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
      y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
      fold_counter += 1
      # Print the current fold and repeat number
      print(f"Repeat: {repeat_counter + 1}, Fold: {fold_counter}",' | alpha:',params)

      # Check if all folds in the current repeat are completed
      if fold_counter == n_splits:
        # Reset the fold counter
        fold_counter = 0

        # Increment the repeat counter
        repeat_counter += 1
      # Create matrix based on word frequency in tweets
      vectorizer = TfidfVectorizer()
      X_train_fold = vectorizer.fit_transform(X_train_fold)
      X_val_fold = vectorizer.transform(X_val_fold)
      X_test_fold = vectorizer.transform(test_set['cleaned_text'])

      # Print the size of our data
      print(f'Training size: {X_train_fold.shape[0]} tweets\t\
      Validation size: {X_val_fold.shape[0]} tweets\t\
      Test size: {X_test_fold.shape[0]} tweets\t\
      Amount of words (columns): {X_train_fold.shape[1]} words')

      # Multinomial Naive Bayes
      alpha_value = params['alphas']
      multi_nb = MultinomialNB(alpha=alpha_value, class_prior=None, fit_prior=True)
      multi_nb.fit(X_train_fold, y_train_fold)

      # Predict on training and validation sets
      train_pred = multi_nb.predict(X_train_fold)
      val_pred = multi_nb.predict(X_val_fold)

      # Calculate and print accuracy on training and validation sets
      train_accuracy = accuracy_score(y_train_fold, train_pred)
      val_accuracy = accuracy_score(y_val_fold, val_pred)
      #print('train_index:',train_index)
      print(f'Accuracy on training set (MultinomialNB): {round(train_accuracy * 100, 4)}%')
      print(f'Accuracy on validation set (MultinomialNB): {round(val_accuracy * 100, 4)}%')

      # Append accuracy score to list
      val_accuracy_scores.append(val_accuracy)
      mean_val_accuracy = np.mean(val_accuracy_scores)
      mean_val_accuracies.append(mean_val_accuracy)

      train_accuracy_scores.append(train_accuracy)
      mean_train_accuracy = np.mean(train_accuracy_scores)
      mean_train_accuracies.append(mean_train_accuracy)
  # Calculate the mean validation accuracy across folds
  mean_val_accuracy = np.mean(val_accuracy_scores)
  mean_train_accuracy = np.mean(train_accuracy_scores)
  print("Mean Validation Accuracy:", mean_val_accuracy)
  print("Mean Training Accuracy:", mean_train_accuracy)
  print("====================================================================================================")
print(" accuracy array:",val_accuracy_scores)

# Find the hyperparameters with the highest mean validation accuracy
best_params_index = np.argmax(mean_val_accuracies)
best_params = list(ParameterGrid(param_grid))[best_params_index]
best_mean_val_accuracy = mean_val_accuracies[best_params_index]

print("Hyperparameters for highest Validation Accuracy:", best_params)
print("Highest Mean Validation Accuracy:", best_mean_val_accuracy)




#Sklearn's Gradient Boosting Machine (GBM)


#Sklearn's Gradient Boosting Classifier (GBM)
gbm = GradientBoostingClassifier(n_estimators=200,
                                 max_depth=6,
                                 random_state=seed)
gbm.fit(X_train_original, y_train)



#Sklearn's Gradient Boosting Machine (GBM) -- Updated Code


# Define the number of splits and repetitions for repeated stratified k-fold cross-validation
n_splits = 10  # Number of folds
n_repeats = 1  # Number of repetitions

# Define the hyperparameter grid for GBM
param_grid = {
    'n_estimators': [300], # Number of trees
    'max_depth': [9],  # Maximum depth of trees
    'learning_rate': [0.1],  # Maximum depth of trees
    'min_samples_leaf': [8]  # Maximum depth of trees

}



# Initialize the repeated stratified k-fold cross-validator
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

# Lists to store accuracy scores for each fold
mean_train_accuracies = []
mean_val_accuracies = []

# Iterate over the hyperparameter grid
for params in ParameterGrid(param_grid):
    train_accuracy_scores = []
    val_accuracy_scores = []
    fold_counter = 0
    repeat_counter = 0

    # Iterate over the folds
    for train_index, val_index in rkf.split(X_train, y_train):
        # Get training and validation data for this fold
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        fold_counter += 1

        # Print the current fold and repeat number
        print(f"Repeat: {repeat_counter + 1}, Fold: {fold_counter}", ' | Hyperparameters:', params)
        # Create matrix based on word frequency in tweets
        vectorizer = TfidfVectorizer()
        X_train_fold = vectorizer.fit_transform(X_train_fold)
        X_val_fold = vectorizer.transform(X_val_fold)
        X_test_fold = vectorizer.transform(test_set['cleaned_text'])

        # Print the size of our data
        print(f'Training size: {X_train_fold.shape[0]} tweets\t\
        Validation size: {X_val_fold.shape[0]} tweets\t\
        Test size: {X_test_fold.shape[0]} tweets\t\
        Amount of words (columns): {X_train_fold.shape[1]} words')

        # Create and fit the GBM model
        n_estimators_value = params['n_estimators']
        max_depth_value = params['max_depth']
        learning_rate_value = params['learning_rate']
        min_samples_leaf_value=params['min_samples_leaf']
        gbm = GradientBoostingClassifier(n_estimators=n_estimators_value,
                                 max_depth=max_depth_value,
                                  learning_rate=learning_rate_value,
                                  max_features='sqrt',
                                  min_samples_leaf=min_samples_leaf_value,
                                 random_state=seed)
        gbm.fit(X_train_fold, y_train_fold)

        # Predict on training and validation sets
        train_pred = gbm.predict(X_train_fold)
        val_pred = gbm.predict(X_val_fold)

        # Calculate and print accuracy on training and validation sets
        train_accuracy = accuracy_score(y_train_fold, train_pred)
        val_accuracy = accuracy_score(y_val_fold, val_pred)
        print(f'Accuracy on training set (GBM): {round(train_accuracy * 100, 4)}%')
        print(f'Accuracy on validation set (GBM): {round(val_accuracy * 100, 4)}%')

        # Append accuracy score to list
        train_accuracy_scores.append(train_accuracy)
        val_accuracy_scores.append(val_accuracy)

        # Check if all folds in the current repeat are completed
        if fold_counter == n_splits:
            # Reset the fold counter
            fold_counter = 0
            # Increment the repeat counter
            repeat_counter += 1

    # Calculate the mean training accuracy across folds
    mean_train_accuracy = np.mean(train_accuracy_scores)
    mean_train_accuracies.append(mean_train_accuracy)
    print("Mean Training Accuracy:", mean_train_accuracy)
    print("====================================================================================================")

    # Calculate the mean validation accuracy across folds
    mean_val_accuracy = np.mean(val_accuracy_scores)
    mean_val_accuracies.append(mean_val_accuracy)
    print("Mean Validation Accuracy:", mean_val_accuracy)
    print("====================================================================================================")

# Find the hyperparameters with the highest mean validation accuracy
best_params_index = np.argmax(mean_val_accuracies)
best_params = list(ParameterGrid(param_grid))[best_params_index]
best_mean_val_accuracy = mean_val_accuracies[best_params_index]

print("Best Hyperparameters:", best_params)
print("Best Mean Validation Accuracy:", best_mean_val_accuracy)




#XGBoost (GBM)


# Hyperparameters that you can tweak
# There are a lot more tweakable hyperparameters that you can find at
# https://xgboost.readthedocs.io/en/latest/parameter.html
xgb_params = {'objective' : 'multi:softmax',
              'eval_metric' : 'mlogloss',
              'eta' : 0.1,
              'max_depth' : 6,
              'num_class' : 3,
              'lambda' : 0.8,
              'estimators' : 200,
              'seed' : seed

}

# Transform categories into numbers
# negative = 0, neutral = 1 and positive = 2
target_train = y_train.astype('category').cat.codes
target_val = y_val.astype('category').cat.codes

# Transform data into a matrix so that we can use XGBoost
d_train = xgb.DMatrix(X_train_original, label = target_train)
d_val = xgb.DMatrix(X_val_original, label = target_val)

# Fit XGBoost
watchlist = [(d_train, 'train'), (d_val, 'validation')]
bst = xgb.train(xgb_params,
                d_train,
                400,
                watchlist,
                early_stopping_rounds = 50,
                verbose_eval = 0)



# Setting parameters for XGBoost
xgb_params = {
    'objective': 'multi:softmax',
    'max_depth': 6,
    'num_class': 3,
    'lambda': 0.8,
    'seed': seed
}

# Transform categories into numbers
# negative = 0, neutral = 1, positive = 2
target_train = y_train.astype('category').cat.codes
target_val = y_val.astype('category').cat.codes

# Transform data into a matrix so that we can use XGBoost
d_train = xgb.DMatrix(X_train_original, label=target_train)
d_val = xgb.DMatrix(X_val_original, label=target_val)

# Fit XGBoost
watchlist = [(d_train, 'train'), (d_val, 'validation')]
bst = xgb.train(params=xgb_params,
                dtrain=d_train,
                num_boost_round=200,  # Here we specify the number of boosting rounds instead of 'estimators'
                evals=watchlist,  # Specify evals using keyword
                early_stopping_rounds=50,
                verbose_eval=0)




print("X_train_original shape:", X_train_original.shape)
print("target_train shape:", target_train.shape)
print("X_val_original shape:", X_val_original.shape)
print("target_val shape:", target_val.shape)



#XGBoost (GBM) -- Upgraded


# Define the number of splits and repetitions for repeated stratified k-fold cross-validation
n_splits = 3  # Number of folds
# n_splits = 2  # Number of folds
n_repeats = 1  # Number of repetitions

# Define the hyperparameter grid for XGBoost
param_grid = {
    'max_depth': [9],         # Maximum depth of trees
    'eta': [0.1],   # Learning rate
    'objective' : ['multi:softmax'],
    'eval_metric' : ['mlogloss'],
    'eta' : [0.1],
    'num_class' : [3],
    'lambda' : [0.1]

}



# Initialize the repeated stratified k-fold cross-validator
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

# Lists to store accuracy scores for each fold
mean_val_accuracies = []

# Iterate over the hyperparameter grid
for params in ParameterGrid(param_grid):
    val_accuracy_scores = []
    fold_counter = 0
    repeat_counter = 0

    # Iterate over the folds
    for train_index, val_index in rkf.split(X_train, y_train):
        # Get training and validation data for this fold
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        fold_counter += 1

        # Print the current fold and repeat number
        print(f"Repeat: {repeat_counter + 1}, Fold: {fold_counter}", ' | Hyperparameters:', params)

        # Create matrix based on word frequency in tweets
        vectorizer = TfidfVectorizer()
        X_train_fold = vectorizer.fit_transform(X_train_fold)
        X_val_fold = vectorizer.transform(X_val_fold)
        X_test_fold = vectorizer.transform(test_set['cleaned_text'])

        # Print the size of our data
        print(f'Training size: {X_train_fold.shape[0]} tweets\t\
        Validation size: {X_val_fold.shape[0]} tweets\t\
        Test size: {X_test_fold.shape[0]} tweets\t\
        Amount of words (columns): {X_train_fold.shape[1]} words')

        # Create and fit the XGBM model

        # Transform categories into numbers
        # negative = 0, neutral = 1 and positive = 2
        target_train = y_train_fold.astype('category').cat.codes
        target_val = y_val_fold.astype('category').cat.codes

        # Transform data into a matrix so that we can use XGBoost
        d_train = xgb.DMatrix(X_train_fold, label = target_train)
        d_val = xgb.DMatrix(X_val_fold, label = target_val)
        watchlist = [(d_train, 'train'), (d_val, 'validation')]

        bst = xgb.train(params=params,
                        dtrain=d_train,
                        num_boost_round=400,
                        evals=watchlist,
                        early_stopping_rounds=50,
                        verbose_eval=0)

        # Check results for XGBoost
        train_pred = bst.predict(d_train)
        val_pred = bst.predict(d_val)
        print(f'Accuracy on training set (XGBoost): {round(accuracy_score(target_train, train_pred)*100, 4)}%')
        print(f'Accuracy on validation set (XGBoost): {round(accuracy_score(target_val, val_pred)*100, 4)}%')

        # Append accuracy score to list
        val_accuracy_scores.append(val_accuracy)

        # Check if all folds in the current repeat are completed
        if fold_counter == n_splits:
            # Reset the fold counter
            fold_counter = 0
            # Increment the repeat counter
            repeat_counter += 1

    # Calculate the mean validation accuracy across folds
    mean_val_accuracy = np.mean(val_accuracy_scores)
    mean_val_accuracies.append(mean_val_accuracy)
    print("Mean Validation Accuracy:", mean_val_accuracy)
    print("====================================================================================================")
    
    
    
    # Find the hyperparameters with the highest mean validation accuracy
best_params_index = np.argmax(mean_val_accuracies)
best_params = list(ParameterGrid(param_grid))[best_params_index]
best_mean_val_accuracy = mean_val_accuracies[best_params_index]

print("Best Hyperparameters:", best_params)
print("Best Mean Validation Accuracy:", best_mean_val_accuracy)



#Simple neural network


# Generator so we can easily feed batches of data to the neural network
def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = X.shape[0]/batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0



# Onehot encoding of target variable
# Negative = [1,0,0], Neutral = [0,1,0], Positive = [0,0,1]

# Initialize sklearn's one-hot encoder class
onehot_encoder = OneHotEncoder(sparse_output=False)

# One hot encoding for training set
integer_encoded_train = np.array(y_train).reshape(len(y_train), 1)
onehot_encoded_train = onehot_encoder.fit_transform(integer_encoded_train)

# One hot encoding for validation set
integer_encoded_val = np.array(y_val).reshape(len(y_val), 1)
onehot_encoded_val = onehot_encoder.fit_transform(integer_encoded_val)


# Neural network architecture
initializer = keras.initializers.he_normal(seed=seed)
activation = keras.activations.elu
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#optimizer = keras.optimizers.SGD(learning_rate=0.02, momentum=0.9)
#optimizer = keras.optimizers.Adamax(learning_rate=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
#optimizer = keras.optimizers.RMSprop(learning_rate=0.0002, rho=0.99, epsilon=1e-8)

es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=4)

# Build model architecture
model = Sequential()
model.add(Dense(20, activation=activation, kernel_initializer=initializer, input_dim=X_train_original.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax', kernel_initializer=initializer))
model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Hyperparameters
epochs = 50
batch_size = 32


# Convert sparse matrices to dense arrays
X_train_dense = X_train_original.toarray()
X_val_dense = X_val_original.toarray()

# Convert NumPy arrays to TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_dense, onehot_encoded_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)

# Fit the model using the dataset
hist = model.fit(train_dataset,
                 epochs=epochs,
                 validation_data=(X_val_dense, onehot_encoded_val),
                 steps_per_epoch=X_train_original.shape[0] // batch_size,
                 callbacks=[es],shuffle=True)



#Simple neural network- Updated


# Constants
seed = 42
n_splits = 3
n_repeats = 1
epochs = 50
batch_size = 128

# Shuffle dataframe
df = shuffle(df, random_state=seed)
# Keep 1000 samples of the data as test set
test_set = df[:1000]
df_train_val = df[1000:]

# Get sentiment labels for test set
y_test = test_set['airline_sentiment']

# Create matrix based on word frequency in tweets
vectorizer = TfidfVectorizer()
X_test_fold = vectorizer.fit_transform(test_set['cleaned_text'])

# Define cross-validator
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

# Initialize results storage
train_scores = []
val_scores = []

best_fold_history = {'accuracy': [], 'val_accuracy': []}  # Track history for the best fold

# Iterate over each fold
for fold, (train_index, val_index) in enumerate(rkf.split(df_train_val['cleaned_text'], df_train_val['airline_sentiment'])):
    print(f"Fold {fold + 1}:")
    X_train, X_val = df_train_val['cleaned_text'].iloc[train_index], df_train_val['cleaned_text'].iloc[val_index]
    y_train, y_val = df_train_val['airline_sentiment'].iloc[train_index], df_train_val['airline_sentiment'].iloc[val_index]

    # Vectorize text
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)

    # Onehot encoding of target variable
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = onehot_encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
    y_val_encoded = onehot_encoder.transform(y_val.to_numpy().reshape(-1, 1))

    # Neural network architecture
    model = Sequential([
        Dense(20, activation='elu', kernel_initializer='he_normal', input_dim=X_train.shape[1]),
        Dropout(0.5),
        Dense(3, activation='softmax', kernel_initializer='he_normal')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Early stopping
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, restore_best_weights=True)

    # Train the model with early stopping
    hist = model.fit(X_train.toarray(), y_train_encoded,
                     validation_data=(X_val.toarray(), y_val_encoded),
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=[early_stopping],
                     verbose=0)  # Turn off verbosity for cleaner output

    # Store history for this fold
    fold_history = {'accuracy': hist.history['accuracy'], 'val_accuracy': hist.history['val_accuracy']}
    # Append fold history to the overall history
    best_fold_history['accuracy'].append(fold_history['accuracy'])
    best_fold_history['val_accuracy'].append(fold_history['val_accuracy'])

    # Output epoch-wise accuracy
    for epoch in range(len(hist.history['accuracy'])):
        print(f"Epoch {epoch + 1} - Train Accuracy: {hist.history['accuracy'][epoch]}, Validation Accuracy: {hist.history['val_accuracy'][epoch]}")

    # Store scores
    train_scores.append(hist.history['accuracy'][-1])
    val_scores.append(hist.history['val_accuracy'][-1])



#Rule based models


#Always predict negative


# Predict negative for the whole dataset
negative_pred_train = ['negative' for _ in range(len(y_train))]
negative_pred_val = ['negative' for _ in range(len(y_val))]

print(f'Accuracy on training set (Always Predict Negative): {round(accuracy_score(y_train, negative_pred_train)*100, 4)}%')
print(f'Accuracy on validation set (Always Predict Negative): {round(accuracy_score(y_val, negative_pred_val)*100, 4)}%')
"""