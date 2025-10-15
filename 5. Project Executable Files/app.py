from flask import Flask, render_template, request
import re
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

app = Flask(__name__)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/submit', methods=['POST'])
def submit():
    content = request.form.get('content', '')
    if not content:
        return "No input provided", 400

    cleaned = remove_stopwords(clean_text(content))
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    return render_template('output.html', content=content, result=prediction)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
