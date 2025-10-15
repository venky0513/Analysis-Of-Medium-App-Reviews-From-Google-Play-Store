import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
df = pd.read_csv("dataset.csv")

# Fill missing values for consistency
for col in ['reviewCreatedVersion', 'replyContent', 'repliedAt', 'appVersion']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Keep only useful columns
df = df[["content", "sentiment"]]

# -------------------------------
# 2️⃣ Clean and preprocess text
# -------------------------------
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

df['cleaned'] = df['content'].apply(lambda x: remove_stopwords(clean_text(x)))

# -------------------------------
# 3️⃣ Split and vectorize
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['sentiment'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)

# -------------------------------
# 4️⃣ Train model
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# -------------------------------
# 5️⃣ Save model and vectorizer
# -------------------------------
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
