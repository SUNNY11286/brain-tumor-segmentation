from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the trained model
# model = load_model('model.h5')

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image_path):
    image = np.load(image_path)
    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)
    target_shape = (64, 64, 64, 1)
    image = np.pad(image, ((0, max(0, target_shape[0] - image.shape[0])),
                           (0, max(0, target_shape[1] - image.shape[1])),
                           (0, max(0, target_shape[2] - image.shape[2])),
                           (0, 0)), mode='constant')
    image = image[:target_shape[0], :target_shape[1], :target_shape[2], :]
    return np.expand_dims(image, axis=0)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            return redirect(url_for('predict', filename=filename))
    return render_template('upload.html')

@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = preprocess_image(file_path)
    # prediction = model.predict(image)
    # prediction = (prediction > 0.5).astype(np.uint8)  # Assuming binary segmentation

    # # Save the prediction mask
    # mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
    # np.save(mask_path, prediction[0])

    return render_template('result.html', original_image=filename, mask_image='mask_' + filename)

if __name__ == '__main__':
    app.run(debug=True)