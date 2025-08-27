# Lung-Cancer-Detection-using-CT
## Project Overview

This project builds an easy-to-use deep learning model that classifies lung CT scan images into Benign, Malignant, or Normal categories. The model uses a transfer learning approach with EfficientNetB0 to achieve accurate predictions with less training time. Accompanied by a simple Flask web app, users can upload images and get quick diagnostic results.

## Usage

1. Run the Flask web app:

python app.py

2. Open your browser and go to `http://127.0.0.1:5000/`.


3. Use the web interface to upload lung CT scan images (JPEG/PNG).

4. The app will display the predicted class:
   - Benign
   - Malignant
   - Normal

5. View confidence scores indicating prediction certainty.
## Features

Classifies lung CT scans into Benign, Malignant, or Normal with high accuracy.

Utilizes transfer learning with the EfficientNetB0 architecture for efficient training.

Simple Flask web application interface for easy image upload and instant prediction.

Displays confidence scores alongside predicted classes to help assess certainty.

Open-source and extendable for further development and improvements.
