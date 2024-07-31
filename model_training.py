# model_training.py
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd
from preprocessing import vectorizer, X_train_vect, X_test_vect, y_train, y_test

# Initialize and train the model
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_vect)
print(classification_report(y_test, y_pred))
