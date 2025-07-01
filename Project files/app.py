from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained model and labels once
model = load_model('backend/model/poultry_model.h5')
labels = open('backend/model/poultry_labels.txt').read().splitlines()

# Prediction function
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to model input
    img_array = image.img_to_array(img) / 255.0              # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)[0]
    index = np.argmax(predictions)
    return labels[index], round(predictions[index] * 100, 2)

# Home page
@app.route('/')
def index():
    return render_template('index.html')

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No image uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction="No file selected")

    # Save uploaded file to static/
    filepath = os.path.join('static', file.filename)
    os.makedirs('static', exist_ok=True)
    file.save(filepath)

    # Predict
    label, confidence = model_predict(filepath)

    return render_template(
        'index.html',
        prediction=label,
        confidence=confidence,
        image_url=filepath
    )

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
