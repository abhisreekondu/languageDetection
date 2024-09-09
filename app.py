# app.py
from flask import Flask, render_template, request, jsonify
from model import predict_language  # Import your prediction function

app = Flask(__name__)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']  # Get the text input from form
        prediction = predict_language(text)  # Call the model's predict function
        return render_template('index.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
