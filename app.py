from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model & scaler
model = joblib.load('heart_disease_model1.pkl')
scaler = joblib.load('scaler1.pkl')


@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/result_view')
def result_view():
    return render_template('result_view.html')

@app.route('/about')
def about():
    return render_template('about.html') 

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1, -1)

    # Scale input
    features = scaler.transform(features)

    # Predict
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction] * 100  # Get confidence score

    # Determine result message
    if prediction == 1:
        result = "Heart Disease Detected"
        color = "red"
    else:
        result = "No Heart Disease"
        color = "green"

    return render_template('result.html', prediction_text=result, confidence=confidence, color=color)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

