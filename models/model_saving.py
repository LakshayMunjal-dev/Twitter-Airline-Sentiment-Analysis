# Save XGBoost model
pickle.dump(bst, open('xgboost_sentiment_model.dat', 'wb'))

# Save Neural Network model
model.save('nn_sentiment_model.h5')