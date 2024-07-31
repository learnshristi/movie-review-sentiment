# cli.py
from preprocessing import vectorizer
from model_training import model

def predict_sentiment(review):
    review_vect = vectorizer.transform([review])
    prediction = model.predict(review_vect)
    return prediction[0]

if __name__ == '__main__':
    review = input("Enter a movie review: ")
    sentiment = predict_sentiment(review)
    print(f"The sentiment of the review is: {sentiment}")
