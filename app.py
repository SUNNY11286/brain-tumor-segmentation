from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import numpy as np
import os
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import shutil


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['STATIC_FOLDER'] = os.path.join('static', 'uploads')

# Load the trained model
model = load_model('model.h5')

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

# def combine_slices(image, method='grid'):
#     """
#     Combine slices from a 3D or 4D image into a single 2D image.
#     """
#     # Handle 4D arrays (e.g., multiple channels or batches)
#     if image.ndim == 4:
#         if image.shape[-1] == 1:
#             image = image[..., 0]  # Remove single channel dimension
#         else:
#             image = np.mean(image, axis=-1)  # Average over channels/batch

#     # Now, the image should be 3D (height, width, depth)
#     if image.ndim != 3:
#         raise ValueError(f"Expected a 3D array after processing, but got shape: {image.shape}")
    
#     height, width, depth = image.shape

#     # Concatenate slices into a grid
#     grid_size = int(np.ceil(np.sqrt(depth)))
    
#     # Create an empty array to hold the grid, with appropriate padding if necessary
#     grid_image = np.zeros((grid_size * height, grid_size * width))
    
#     # Fill the grid with slices
#     for i in range(depth):
#         row = i // grid_size
#         col = i % grid_size
#         slice_img = image[:, :, i]
#         # Place the slice in the correct position in the grid
#         grid_image[row*height:(row+1)*height, col*width:(col+1)*width] = slice_img
    
#     return grid_image


def combine_slices(image, method='grid'):
    """
    Combine slices from a 3D or 4D image into a single 2D image.
    """
    # Handle 4D arrays (e.g., multiple channels or batches)
    if image.ndim == 4:
        if image.shape[-1] == 1:
            image = image[..., 0]  # Remove single channel dimension
        else:
            image = np.mean(image, axis=-1)  # Average over channels/batch

    # Now, the image should be 3D (height, width, depth)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D array after processing, but got shape: {image.shape}")
    
    height, width, depth = image.shape

    if method == 'average':
        # Average across the depth axis
        return np.mean(image, axis=2)
    
    elif method == 'grid':
        # Concatenate slices into a grid
        grid_size = int(np.ceil(np.sqrt(depth)))
        
        # Create an empty array to hold the grid, with appropriate padding if necessary
        grid_image = np.zeros((grid_size * height, grid_size * width))
        
        # Fill the grid with slices
        for i in range(depth):
            row = i // grid_size
            col = i % grid_size
            slice_img = image[:, :, i]
            # Place the slice in the correct position in the grid
            grid_image[row*height:(row+1)*height, col*width:(col+1)*width] = slice_img
        
        return grid_image
    
    else:
        raise ValueError("Invalid method. Choose from 'average' or 'grid'.")


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



# @app.route('/', methods=['GET', 'POST'])
# def upload_file():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files['file']
#         if file.filename == '':
#             return redirect(request.url)
#         if file:
#             filename = secure_filename(file.filename)
#             file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#             file.save(file_path)
#             return redirect(url_for('predict', filename=filename))
#     return render_template('upload.html')

# @app.route('/predict/<filename>')
# def predict(filename):
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     image = preprocess_image(file_path)
#     prediction = model.predict(image)
    
#     # Handle the model output
#     bridge_output, logits = prediction
    
#     # Use the logits (second output) for visualization
#     binary_prediction = (logits[0, ..., 0] > 0.5).astype(np.uint8)
    
#     print("---------- predicted ----------")
#     print("Prediction shape:", binary_prediction.shape)

#     # Save the prediction mask
#     mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'mask_' + filename)
#     np.save(mask_path, binary_prediction)

#     # Visualize original image and prediction
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
#     # Original image
#     original_grid = combine_slices(image[0, ..., 0])
#     ax1.imshow(original_grid, cmap='gray')
#     ax1.set_title('Original Image')
#     ax1.axis('off')
    
#     # Predicted mask
#     prediction_grid = combine_slices(binary_prediction)
#     ax2.imshow(prediction_grid, cmap='gray')
#     ax2.set_title('Predicted Mask')
#     ax2.axis('off')
    
#     plt.tight_layout()
    
#     # Save the plot to a BytesIO object
#     img_buffer = BytesIO()
#     plt.savefig(img_buffer, format='png', bbox_inches='tight')
#     img_buffer.seek(0)
#     img_str = base64.b64encode(img_buffer.getvalue()).decode()
#     plt.close()

#     return render_template('result.html', original_image=filename, mask_image='mask_' + filename, prediction_image=img_str)

# if __name__ == '__main__':
#     app.run(debug=True)



def copy_to_static(filename):
    """Copy uploaded file to static folder for display"""
    src = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    dst = os.path.join(app.config['STATIC_FOLDER'], filename)
    shutil.copy(src, dst)

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
            copy_to_static(filename)  # Copy file to static folder
            return redirect(url_for('predict', filename=filename))
    return render_template('upload.html')


@app.route('/predict/<filename>')
def predict(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = preprocess_image(file_path)
    prediction = model.predict(image)
    
    # Handle the model output
    bridge_output, logits = prediction
    
    # Use the logits (second output) for visualization
    binary_prediction = (logits[0, ..., 0] > 0.5).astype(np.uint8)
    
    print("---------- predicted ----------")
    print("Prediction shape:", binary_prediction.shape)

    # Save the prediction mask
    mask_filename = 'mask_' + filename
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
    np.save(mask_path, binary_prediction)
    copy_to_static(mask_filename)  # Copy mask to static folder

    # Visualize original image and prediction using both grid and average methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Grid method
    original_grid = combine_slices(image[0, ..., 0], method='grid')
    axes[0, 0].imshow(original_grid, cmap='gray')
    axes[0, 0].set_title('Original Image (Grid)')
    axes[0, 0].axis('off')
    
    prediction_grid = combine_slices(binary_prediction, method='grid')
    axes[0, 1].imshow(prediction_grid, cmap='gray')
    axes[0, 1].set_title('Predicted Mask (Grid)')
    axes[0, 1].axis('off')
    
    # Average method
    original_avg = combine_slices(image[0, ..., 0], method='average')
    axes[1, 0].imshow(original_avg, cmap='gray')
    axes[1, 0].set_title('Original Image (Average)')
    axes[1, 0].axis('off')
    
    prediction_avg = combine_slices(binary_prediction, method='average')
    axes[1, 1].imshow(prediction_avg, cmap='gray')
    axes[1, 1].set_title('Predicted Mask (Average)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the plot to a BytesIO object
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()

    return render_template('result.html', original_image=filename, mask_image=mask_filename, prediction_image=img_str)

@app.route('/static/uploads/<filename>')
def serve_file(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)


# @app.route('/predict/<filename>')
# def predict(filename):
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#     image = preprocess_image(file_path)
#     prediction = model.predict(image)
    
#     # Handle the model output
#     bridge_output, logits = prediction
    
#     # Use the logits (second output) for visualization
#     binary_prediction = (logits[0, ..., 0] > 0.5).astype(np.uint8)
    
#     print("---------- predicted ----------")
#     print("Prediction shape:", binary_prediction.shape)

#     # Save the prediction mask
#     mask_filename = 'mask_' + filename
#     mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
#     np.save(mask_path, binary_prediction)
#     copy_to_static(mask_filename)  # Copy mask to static folder

#     # Visualize original image and prediction
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
#     # Original image
#     original_grid = combine_slices(image[0, ..., 0])
#     ax1.imshow(original_grid, cmap='gray')
#     ax1.set_title('Original Image')
#     ax1.axis('off')
    
#     # Predicted mask
#     prediction_grid = combine_slices(binary_prediction)
#     ax2.imshow(prediction_grid, cmap='gray')
#     ax2.set_title('Predicted Mask')
#     ax2.axis('off')
    
#     plt.tight_layout()
    
#     # Save the plot to a BytesIO object
#     img_buffer = BytesIO()
#     plt.savefig(img_buffer, format='png', bbox_inches='tight')
#     img_buffer.seek(0)
#     img_str = base64.b64encode(img_buffer.getvalue()).decode()
#     plt.close()

#     return render_template('result.html', original_image=filename, mask_image=mask_filename, prediction_image=img_str)

# @app.route('/static/uploads/<filename>')
# def serve_file(filename):
#     return send_from_directory(app.config['STATIC_FOLDER'], filename)

# if __name__ == '__main__':
#     app.run(debug=True)
