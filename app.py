from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Training Data
training_sentences = [
    "hello","hi","hey","good morning","good evening",
    "what courses do you offer","tell me about courses","course details","what are your courses","do you have python course",
    "do you provide internship","internship details","internship available","i want internship","tell me about internship",
    "what is fees","course fees","how much fees","fees for python course","tell me fee structure"
]

training_labels = [
    "greeting","greeting","greeting","greeting","greeting",
    "courses","courses","courses","courses","courses",
    "internship","internship","internship","internship","internship",
    "fees","fees","fees","fees","fees"
]

# Model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(training_sentences)

model = MultinomialNB()
model.fit(X, training_labels)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Chat API
@app.route("/chat", methods=["POST"])
def chat():
    user = request.json["message"].lower()

    user_vec = vectorizer.transform([user])
    probs = model.predict_proba(user_vec)
    confidence = max(probs[0])
    prediction = model.predict(user_vec)[0]

    if confidence < 0.4:
        return jsonify({"response": "I didn't understand. Please rephrase."})

    if prediction == "greeting":
        reply = "Hello! Welcome to Electrobtech!"

    elif prediction == "courses":
        reply = "We offer AI, Data Science, Python, IoT."

    elif prediction == "internship":
        reply = "Yes! We provide AI/ML internships."

    elif prediction == "fees":
        reply = "Please contact admin for fee details."

    return jsonify({"response": reply})

import os

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))