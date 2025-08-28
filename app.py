# app.py

from flask import Flask, request, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

app = Flask(__name__)

# Load the trained model
model = load_model('lung_cancer_model.h5')

# Class names exactly as in training
class_names = ['Benign', 'Malignant', 'Normal']

def prepare_image(image_path):
    """
    Load and preprocess the image to be compatible with EfficientNet input.
    """
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)  # Use EfficientNet preprocessing
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    confidence = 0.0
    if request.method == 'POST':
        if 'file' not in request.files:
            prediction = 'No file part'
        else:
            file = request.files['file']
            if file.filename == '':
                prediction = 'No selected file'
            else:
                # Save the uploaded file temporarily
                upload_path = os.path.join('uploads', file.filename)
                os.makedirs('uploads', exist_ok=True)
                file.save(upload_path)

                # Prepare image and predict
                img = prepare_image(upload_path)
                preds = model.predict(img)
                pred_class = np.argmax(preds[0])
                confidence = preds[0][pred_class] * 100
                prediction = class_names[pred_class]

                # Optionally remove file after prediction
                os.remove(upload_path)

    return render_template('index.html', prediction=prediction, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
