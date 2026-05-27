from flask import Flask, render_template, request
import pickle
import string
from nltk.corpus import stopwords

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Text cleaning
def clean_text(text):

    text = text.lower()

    text = ''.join(
        [char for char in text if char not in string.punctuation]
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stopwords.words('english')
    ]

    return ' '.join(words)

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    cleaned = clean_text(message)

    vector_input = vectorizer.transform([cleaned])

    prediction = model.predict(vector_input)[0]

    if prediction == 1:

        result = "🚫 SPAM MESSAGE"

        category = "spam"

    else:

        result = "✅ HAM MESSAGE"

        category = "ham"

    return render_template(
        'index.html',
        prediction=result,
        category=category
    )

if __name__ == '__main__':
    app.run(debug=True)