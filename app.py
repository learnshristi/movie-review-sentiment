from flask import Flask, request, jsonify, render_template
from preprocessing import vectorizer
from model_training import model

app = Flask(__name__)
# app = Flask(__name__, template_folder='path/to/templates')

@app.route('/')
def index():
    return render_template('index.html')

def predict_sentiment(review):
    review_vect = vectorizer.transform([review])
    prediction = model.predict(review_vect)
    return prediction[0]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('review')
    if not data:
        return jsonify({'error': 'No review provided'}), 400
    sentiment = predict_sentiment(data)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, port=5001)
