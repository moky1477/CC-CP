from flask import Flask, render_template, request
import numpy as np
import joblib
import pickle
import os

app = Flask(__name__)

# Load models
def load_ml_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

svm_model = load_ml_model('Models/SVMClassifier.pkl')
naive_bayes_model = load_ml_model('Models/NBClassifier.pkl')
dl_model = load_ml_model('Models/DeepLearning.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['pH Level'])
        rainfall = float(request.form['Rainfall'])
        model_option = request.form['model_option']

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        if model_option == 'Naive Bayes (Accuracy: 99.01%)':
            prediction = naive_bayes_model.predict(input_data)
        elif model_option == 'SVM (Accuracy: 98%)':
            prediction = svm_model.predict(input_data)
        elif model_option == 'Neural Network (Accuracy: 96%)':
            prediction = dl_model.predict(input_data)

        crop_name = prediction[0]
        # input_prompt = """
        # As an experienced farmer with in-depth knowledge of various crops, your expertise is sought by a beginner farmer looking to cultivate a specific crop. Provide detailed guidance on the chosen crop, covering the following aspects:

        # 1. Briefly outline the basic steps, advantages, and applications of the selected crop to help the user understand its significance in farming.
        # 2. Share specific techniques tailored for a beginner farmer to successfully grow the mentioned crop.
        # 3. Ensure the information provided is accurate, avoiding any misleading or incorrect advice.

        # Your goal is to empower the user with valuable insights and practical tips to foster a successful farming experience.
        # """
        # # response = get_gemini_response(input_prompt, crop_name)

        return render_template('result.html', crop_name=crop_name)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
