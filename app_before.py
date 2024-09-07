from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import math

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Load the trained model
model = load_model('model.h5')

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
    prediction = model.predict(image)
    
    # Flatten the prediction
    # flattened_prediction = flatten_nested_list(prediction)
    prediction_array = np.array(prediction)
    
    # Reshape the prediction to a 2D image
    side_length = int(math.sqrt(prediction_array.size))
    binary_prediction = (prediction_array > 0.5).astype(np.uint8).reshape(side_length, -1)
    
    print("---------- predicted ----------")
    print("Prediction shape:", binary_prediction.shape)

    # Save the prediction mask
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
    np.save(mask_path, binary_prediction)

    # Create a plot of the prediction
    plt.figure(figsize=(10, 10))
    plt.imshow(binary_prediction, cmap='gray')
    plt.axis('off')
    plt.title('Prediction')
    
    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return render_template('result.html', original_image=filename, mask_image='mask_' + filename, prediction_image=img_str)

def flatten_nested_list(nested_list):
    flattened = []
    for item in nested_list:
        if isinstance(item, (list, np.ndarray)):
            flattened.extend(flatten_nested_list(item))
        else:
            flattened.append(item)
    return flattened

if __name__ == '__main__':
    app.run(debug=True)