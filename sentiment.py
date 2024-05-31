# Step 1: Data Collection
import pandas as pd

# Load the dataset
data = pd.read_csv("social_media_data.csv")  # Assuming you have a CSV file with 'text' and 'sentiment' columns

# Step 2: Text Processing
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text.lower())
    # Remove stopwords and lemmatize words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

data['clean_text'] = data['text'].apply(preprocess_text)

# Step 3: Feature Extraction
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment']

# Step 4: Model Selection
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Step 5: Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Step 6: Model Evaluation
from sklearn.metrics import classification_report

y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 7: Deployment (Example: Flask Web Service)
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    text = request.json['text']
    cleaned_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = classifier.predict(text_vectorized)
    return jsonify({'sentiment': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
