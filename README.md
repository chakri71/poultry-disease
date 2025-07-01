# Project Title: Transfer Learning-Based Classification of Poultry Diseases

## 🔍 Problem Statement
Develop an AI-based system to classify poultry diseases (Salmonella, New Castle Disease, Coccidiosis, Healthy) using transfer learning. This project supports veterinary education by providing a tool for image-based disease diagnosis and learning.

## 🧠 Technologies Used
- Python 3.8+
- TensorFlow / Keras (MobileNetV2)
- Flask (Web framework)
- OpenCV, Pillow
- HTML, CSS (Frontend with optional Bootstrap)
- VS Code

## 📁 Project Structure
- backend/model/ → Trained model (.h5) and labels
- backend/dataset/ → Organized training data by class
- templates/ → index.html (prediction UI), about.html
- static/ → style.css and uploaded images
- app.py → Flask backend for predictions

## 📊 Dataset
Manually curated images (Kaggle + Google Images Download):
- Salmonella
- New Castle Disease
- Coccidiosis
- Healthy

## ✅ Results
Achieved 90–95% validation accuracy using MobileNetV2 with frozen base layers and custom dense head.

## 🎥 Demo Video
https://drive.google.com/file/d/1IxGjKdSFVLEbF8t-__tRqIFyNDBuBEL2/view?usp=sharing

## 👩‍💻 Contributors
- Reddy Sai Chakri
