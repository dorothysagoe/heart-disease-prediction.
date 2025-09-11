from flask import Flask, request, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), '../models/best_model.pkl')
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles the form submission and makes a prediction.
    """
    # Get values from the form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Make prediction
    prediction = model.predict(final_features)
    prediction_proba = model.predict_proba(final_features)

    # Map prediction to user-friendly message
    target_names = ['No Heart Disease', 'Heart Disease']
    result = target_names[prediction[0]]
    confidence = round(prediction_proba[0][prediction[0]] * 100, 2)

    return render_template('index.html',
                           prediction_text=f'Prediction: {result}',
                           confidence_text=f'Confidence: {confidence}%')

if __name__ == '__main__':
    app.run(debug=True)
