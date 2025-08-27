from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load your trained lung cancer classification model
model = load_model('lung_cancer_model.h5')

# Classes must match the model training order
class_labels = ['Benign', 'Malignant', 'Normal']

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0  # Normalize pixels
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    image = prepare_image(filepath)
    preds = model.predict(image)
    idx = np.argmax(preds[0])
    confidence = preds[0][idx]

    result = {
        'class': class_labels[idx],
        'confidence': f"{confidence * 100:.2f}%"
    }

    os.remove(filepath)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
