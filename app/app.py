import sys
import os
from flask import Flask, request, render_template, send_from_directory
from PIL import Image

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from upload import upload_image
from transformations import color_change, rotation, cropping, flipping
from filters import sharpen, smooth, custom_filters
from intensity_manipulations import tonal_transformation, color_balancing
from segmentation import region_based, deep_learning_segmentation
from deep_learning import style_transfer, gan_image_generation

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    if file:
        filename = 'uploaded_image.jpg'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Apply transformations and filters
        image = Image.open(filepath)
        
        # Example transformations
        image = color_change.apply(image)
        image = rotation.apply(image)
        image = cropping.apply(image, (10, 10, 200, 200))
        image = flipping.apply(image)
        
        # Example filters
        image = sharpen.apply(image)
        image = smooth.apply(image)
        image = custom_filters.apply(image)
        
        # Example intensity manipulations
        image = tonal_transformation.apply(image)
        image = color_balancing.apply(image)
        
        # Save processed image
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_image.jpg')
        image.save(processed_path)
        
        return send_from_directory(app.config['PROCESSED_FOLDER'], 'processed_image.jpg')

if __name__ == "__main__":
    app.run(debug=True)
