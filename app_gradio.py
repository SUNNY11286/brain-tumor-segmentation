import gradio as gr
import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import shutil

# Load the trained model
model = load_model('model.h5')

# Configure upload and static folders
UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = os.path.join('static', 'uploads')

# Ensure the upload and static folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

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
            grid_image[row*height:(row+1)*height, col*width:(col+1)*width] = image[:, :, i]
        
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

def copy_to_static(filename):
    """Copy uploaded file to static folder for display"""
    src = os.path.join(UPLOAD_FOLDER, filename)
    dst = os.path.join(STATIC_FOLDER, filename)
    shutil.copy(src, dst)

def predict_and_visualize(file_obj):
    # Save the uploaded file to the upload folder
    filename = secure_filename(file_obj.name)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file_obj.save(file_path)
    copy_to_static(filename)  # Copy file to static folder

    # Preprocess the image
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
    mask_path = os.path.join(UPLOAD_FOLDER, mask_filename)
    np.save(mask_path, binary_prediction)
    copy_to_static(mask_filename)  # Copy mask to static folder

    # Visualize original image and prediction using both grid and average methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # Grid method visualization
    original_grid = combine_slices(image[0, ..., 0], method='grid')
    axes[0, 0].imshow(original_grid, cmap='gray')
    axes[0, 0].set_title('Original Image (Grid)')
    axes[0, 0].axis('off')
    
    prediction_grid = combine_slices(binary_prediction, method='grid')
    axes[0, 1].imshow(prediction_grid, cmap='gray')
    axes[0, 1].set_title('Predicted Mask (Grid)')
    axes[0, 1].axis('off')
    
    # Average method visualization
    original_avg = combine_slices(image[0, ..., 0], method='average')
    axes[1, 0].imshow(original_avg, cmap='gray')
    axes[1, 0].set_title('Original Image (Average)')
    axes[1, 0].axis('off')
    
    prediction_avg = combine_slices(binary_prediction, method='average')
    axes[1, 1].imshow(prediction_avg, cmap='gray')
    axes[1, 1].set_title('Predicted Mask (Average)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save the plot to a BytesIO object and encode it as base64
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # Return the image as a data URL
    return f"data:image/png;base64,{img_str}"

def secure_filename(filename):
    """Sanitize the filename to prevent directory traversal attacks"""
    return os.path.basename(filename)

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.File(file_types=[".npy"], label="Upload Numpy Image (.npy)"),
    outputs=gr.Image(label="Prediction Results"),
    title="3D Image Prediction",
    description="Upload a 3D NumPy (.npy) image file to get predictions and visualizations."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
