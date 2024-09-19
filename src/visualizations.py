# Plot sentiment distribution
df['airline_sentiment'].value_counts().plot(kind = 'barh',
                                            figsize = (15,10));
plt.title('Distribution of airline sentiment in Kaggle dataset',
          fontsize = 26, weight = 'bold')
plt.xlabel('Frequency', fontsize = 22)
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20);



#Multinomial Naive Bayes

# Define alpha values
alpha_values = [0.45, 0.58, 0.75, 0.85, 0.93, 1]

# Lists to store accuracy scores
train_accuracies = []
val_accuracies = []

# Train Multinomial Naive Bayes classifier for each alpha value
for alpha in alpha_values:
    multi_nb = MultinomialNB(alpha=alpha, class_prior=None, fit_prior=True)
    multi_nb.fit(X_train_original, y_train)

    # Predictions
    train_pred = multi_nb.predict(X_train_original)
    val_pred = multi_nb.predict(X_val_original)

    # Calculate accuracy scores
    train_accuracy = round(accuracy_score(y_train, train_pred) * 100, 4)
    val_accuracy = round(accuracy_score(y_val, val_pred) * 100, 4)

    # Append accuracy scores to lists
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

# Visualize the accuracy scores for different alpha values
plt.figure(figsize=(10, 6))
plt.plot(alpha_values, train_accuracies, marker='o', label='Training Set')
plt.plot(alpha_values, val_accuracies, marker='o', label='Validation Set')
plt.title('Accuracy vs. Alpha Value (Multinomial Naive Bayes)')
plt.xlabel('Alpha Value')
plt.ylabel('Accuracy (%)')
plt.xticks(alpha_values)
plt.legend()
plt.grid(True)
plt.show()

# Print accuracy on training and validation sets for each alpha value
for i, alpha in enumerate(alpha_values):
    print(f'Alpha: {alpha}, Accuracy on training set: {train_accuracies[i]}%, Accuracy on validation set: {val_accuracies[i]}%')


#Multinomial Naive Bayes -- Updated


Plotting Mean Accuracies vs Alpha Values


# Alpha values and corresponding mean validation accuracies
alphas = [0.45, 0.58, 0.75, 0.85, 0.93, 1]

# Dummy mean validation accuracies (replace with your actual values)
mean_val_accuracies = [0.85, 0.82, 0.88, 0.90, 0.87, 0.84]

# Assuming you have calculated mean_train_accuracies as well
mean_train_accuracies = [0.92, 0.91, 0.94, 0.95, 0.93, 0.91]

# Plot mean accuracy versus alpha values
plt.plot(alphas, mean_val_accuracies, marker='o', label='Validation Accuracy')
plt.plot(alphas, mean_train_accuracies, marker='o', label='Training Accuracy')
plt.title('Mean Accuracies vs. Alpha Values')
plt.xlabel('Alpha Values')
plt.ylabel('Mean Accuracies')
plt.legend()
plt.grid(True)
plt.show()




#Plotting Accuracies vs Alpha Values for the Alpha Values of 0.45


# Define the number of splits and repetitions for repeated stratified k-fold cross-validation
n_splits = 30  # Number of folds
n_repeats = 1  # Number of repetitions
param_grid = {
    'alphas': [0.45, 0.58, 0.75, 0.85, 0.93, 1]
}

# Lists to store accuracy scores for each fold
fold_train_accuracies = {alpha: [] for alpha in param_grid['alphas']}
fold_val_accuracies = {alpha: [] for alpha in param_grid['alphas']}

# Repeated stratified k-fold cross-validator
rkf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

# Iterate over hyperparameters
for params in ParameterGrid(param_grid):
    for train_index, val_index in rkf.split(X_train, y_train):
        # Get training and validation data for this fold
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer()
        X_train_fold = vectorizer.fit_transform(X_train_fold)
        X_val_fold = vectorizer.transform(X_val_fold)

        # Multinomial Naive Bayes
        multi_nb = MultinomialNB(alpha=params['alphas'])
        multi_nb.fit(X_train_fold, y_train_fold)

        # Predict on training and validation sets
        train_pred = multi_nb.predict(X_train_fold)
        val_pred = multi_nb.predict(X_val_fold)

        # Calculate accuracies for this fold
        train_accuracy = accuracy_score(y_train_fold, train_pred)
        val_accuracy = accuracy_score(y_val_fold, val_pred)

        fold_train_accuracies[params['alphas']].append(train_accuracy)
        fold_val_accuracies[params['alphas']].append(val_accuracy)

# Plot training and validation accuracies for each fold at alpha = 0.45
alpha_of_interest = 0.85
plt.figure(figsize=(8, 6))
for i, (train_accuracy, val_accuracy) in enumerate(zip(fold_train_accuracies[alpha_of_interest], fold_val_accuracies[alpha_of_interest])):
    plt.scatter(i+1, train_accuracy, color='blue', label='Training Accuracies' if i == 0 else '')
    plt.scatter(i+1, val_accuracy, color='red', label='Validation Accuracies' if i == 0 else '')

plt.title('Accuracies for Alpha = {}'.format(alpha_of_interest))
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()



#Simple Neural Network


# Visualization
accuracy = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

# Dummy test accuracy for each epoch; usually, this would be done per epoch if available
test_acc_per_epoch = [] * len(accuracy)  # Placeholder if test_acc is constant or aggregated

n_epochs = range(len(accuracy))

# Plot training, validation, and testing accuracy
plt.figure(figsize=(15,5))
plt.plot(n_epochs, accuracy, label='Training Accuracy')
plt.plot(n_epochs, val_acc, label='Validation Accuracy')
plt.title('Accuracy over Epochs', weight='bold', fontsize=22)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.show()


#Simple Neural Network -- Updated


# Find the index where training was terminated
early_stopping_epoch = len(best_fold_history['accuracy'][best_val_accuracy_index])

# Plot accuracy across epochs for the best fold
plt.figure(figsize=(10, 6))
plt.plot(range(1, early_stopping_epoch + 1), best_fold_history['accuracy'][best_val_accuracy_index], label='Train Accuracy', marker='o')
plt.plot(range(1, early_stopping_epoch + 1), best_fold_history['val_accuracy'][best_val_accuracy_index], label='Validation Accuracy', marker='o')
plt.title('Accuracy Across Epochs (Best Fold)')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# Print best validation accuracy and hyperparameters
best_val_accuracy = max(val_scores)
best_val_accuracy_index = val_scores.index(best_val_accuracy)
print(f"Best Validation Accuracy: {best_val_accuracy}")
print(f"Epochs: {early_stopping_epoch}")
print(f"Batch Size: {batch_size}")


