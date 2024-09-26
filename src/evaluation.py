# Check results for Multinomial Naive Bayes

train_pred = multi_nb.predict(X_train_original)
val_pred = multi_nb.predict(X_val_original)
print(f'Accuracy on training set (MultinomialNB): {round(accuracy_score(y_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (MultinomialNB): {round(accuracy_score(y_val,val_pred)*100, 4)}%')


#Sklearn's Gradient Boosting Machine (GBM)


# Check results
train_pred = gbm.predict(X_train_original)
val_pred = gbm.predict(X_val_original)
print(f'Accuracy on training set (GBM): {round(accuracy_score(y_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (GBM): {round(accuracy_score(y_val,val_pred)*100, 4)}%')



#Sklearn's Gradient Boosting Machine (GBM) -- Updated


# Find the hyperparameters with the highest mean validation accuracy
best_params_index = np.argmax(mean_val_accuracies)
best_params = list(ParameterGrid(param_grid))[best_params_index]
best_mean_val_accuracy = mean_val_accuracies[best_params_index]

print("Best Hyperparameters:", best_params)
print("Best Mean Validation Accuracy:", best_mean_val_accuracy)


#XGBoost (GBM)


# Check results for XGBoost
train_pred = bst.predict(d_train)
val_pred = bst.predict(d_val)
print(f'Accuracy on training set (XGBoost): {round(accuracy_score(target_train, train_pred)*100, 4)}%')
print(f'Accuracy on validation set (XGBoost): {round(accuracy_score(target_val, val_pred)*100, 4)}%')



#Simple Neural Network -- Updated

# Average results from all folds
average_train_accuracy = np.mean(train_scores)
average_val_accuracy = np.mean(val_scores)

print(f'Average Training Accuracy: {average_train_accuracy*100:.2f}%')
print(f'Average Validation Accuracy: {average_val_accuracy*100:.2f}%')


