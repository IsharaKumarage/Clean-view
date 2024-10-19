import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageFilter
from matplotlib.image import pil_to_array
import numpy as np
import cv2
from skimage import segmentation, color, feature, filters
from sklearn.cluster import KMeans
import tensorflow as tf

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clean-View Image Processing App")

        # Initialize variables
        self.original_image = None
        self.modified_image = None
        self.cv_image_path = None  # To store the path of the opened image
        self.segmented_image = None

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Create a frame for the image display
        self.image_frame = tk.Frame(self.root, width=400, height=600, bg='lightgray')
        self.image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Create a frame for the settings area
        self.settings_frame = ttk.Notebook(self.root)
        self.settings_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Create tabs
        self.filter_tab = ttk.Frame(self.settings_frame)
        self.transformation_tab = ttk.Frame(self.settings_frame)
        self.color_mode_tab = ttk.Frame(self.settings_frame)
        self.advanced_tab = ttk.Frame(self.settings_frame)
        self.intensity_tab = ttk.Frame(self.settings_frame)  # Tab for intensity manipulation
        self.segmentation_tab = ttk.Frame(self.settings_frame)  # Tab for segmentation

        self.settings_frame.add(self.color_mode_tab, text="Color Modes")
        self.settings_frame.add(self.filter_tab, text="Filters")
        self.settings_frame.add(self.transformation_tab, text="Transformations")
        self.settings_frame.add(self.intensity_tab, text="Intensity Manipulation")
        self.settings_frame.add(self.segmentation_tab, text="Segmentation")
        self.settings_frame.add(self.advanced_tab, text="Style Transfer features")
    

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Create image labels
        self.original_image_label = tk.Label(self.image_frame, text="Original Image", bg="lightgray")
        self.original_image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.modified_image_label = tk.Label(self.image_frame, text="Modified Image", bg="lightgray")
        self.modified_image_label.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Create a Menu Bar
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File Menu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open Image", command=self.open_image)
        self.file_menu.add_command(label="Save Image", command=self.save_image)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.root.quit)

        # Feature settings controls
        self.create_filter_controls()
        self.create_transformation_controls()
        self.create_color_mode_controls()
        self.create_styletransfer_controls()
        self.create_intensity_controls()  # Intensity manipulation controls
        self.create_segmentation_controls()  # Segmentation controls

        # Add Reset Button
        self.reset_button = tk.Button(self.root, text="Reset", command=self.reset_image)
        self.reset_button.grid(row=1, column=0, columnspan=2, pady=10, sticky="ew")

    def create_filter_controls(self):
        # Filters Button
        self.filters_button = tk.Button(self.filter_tab, text="Apply Filters", command=self.apply_filters)
        self.filters_button.grid(row=0, column=0, padx=10, pady=10, columnspan=3)

        # Filter Checkbuttons and Sliders
        self.sharpening_var = tk.BooleanVar()
        self.smoothing_var = tk.BooleanVar()
        self.embossing_var = tk.BooleanVar()
        self.edge_detection_var = tk.BooleanVar()

        tk.Checkbutton(self.filter_tab, text="Sharpening", variable=self.sharpening_var).grid(row=1, column=0, padx=10, pady=5)
        self.sharpening_scale = tk.Scale(self.filter_tab, from_=0.0, to=2.0, orient="horizontal", length=200, resolution=0.1)
        self.sharpening_scale.grid(row=2, column=0, padx=10, pady=5)
        self.sharpening_scale.set(1.0)  # Set default value

        tk.Checkbutton(self.filter_tab, text="Smoothing", variable=self.smoothing_var).grid(row=1, column=1, padx=10, pady=5)
        self.smoothing_scale = tk.Scale(self.filter_tab, from_=0.0, to=10.0, orient="horizontal", length=200, resolution=0.1)
        self.smoothing_scale.grid(row=2, column=1, padx=10, pady=5)
        self.smoothing_scale.set(1.0)  # Set default value

        tk.Checkbutton(self.filter_tab, text="Embossing", variable=self.embossing_var).grid(row=1, column=2, padx=10, pady=5)
        tk.Checkbutton(self.filter_tab, text="Edge Detection", variable=self.edge_detection_var).grid(row=2, column=2, padx=10, pady=5)

    def create_transformation_controls(self):
        # Cropping Button and Entry Fields
        tk.Label(self.transformation_tab, text="Crop Width:").grid(row=1, column=0, padx=10, pady=5)
        self.crop_width_var = tk.IntVar(value=100)
        self.crop_width_entry = tk.Entry(self.transformation_tab, textvariable=self.crop_width_var)
        self.crop_width_entry.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(self.transformation_tab, text="Crop Height:").grid(row=2, column=0, padx=10, pady=5)
        self.crop_height_var = tk.IntVar(value=100)
        self.crop_height_entry = tk.Entry(self.transformation_tab, textvariable=self.crop_height_var)
        self.crop_height_entry.grid(row=2, column=1, padx=10, pady=5)

        self.crop_button = tk.Button(self.transformation_tab, text="Apply Crop", command=self.crop_image)
        self.crop_button.grid(row=3, column=0, padx=10, pady=10, columnspan=2)

        # Flipping Button
        self.flip_button = tk.Button(self.transformation_tab, text="Flip Colors", command=self.flip_colors)
        self.flip_button.grid(row=3, column=2, padx=10, pady=10)

        # Rotation Controls
        # Rotation by 90 degrees
        self.rotate_90_button = tk.Button(self.transformation_tab, text="Rotate 90°", command=lambda: self.rotate_fixed(90))
        self.rotate_90_button.grid(row=4, column=0, padx=10, pady=10)

        # Rotation by 180 degrees
        self.rotate_180_button = tk.Button(self.transformation_tab, text="Rotate 180°", command=lambda: self.rotate_fixed(180))
        self.rotate_180_button.grid(row=4, column=1, padx=10, pady=10)

        # Rotation Slider for custom rotation
        self.custom_rotation_scale = tk.Scale(self.transformation_tab, from_=-180, to=180, orient="horizontal", label="Custom Rotation")
        self.custom_rotation_scale.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        self.custom_rotation_button = tk.Button(self.transformation_tab, text="Apply Custom Rotation", command=self.apply_custom_rotation)
        self.custom_rotation_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    def create_color_mode_controls(self):
        # Color Mode Buttons
        self.grayscale_button = tk.Button(self.color_mode_tab, text="Convert to Grayscale", command=self.convert_to_grayscale)
        self.grayscale_button.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.rgb_button = tk.Button(self.color_mode_tab, text="Convert to RGB", command=self.convert_to_rgb)
        self.rgb_button.grid(row=1, column=0, padx=10, pady=10, columnspan=2)

        self.hsv_button = tk.Button(self.color_mode_tab, text="Convert to HSV", command=self.convert_to_hsv)
        self.hsv_button.grid(row=2, column=0, padx=10, pady=10, columnspan=2)

        self.cmyk_button = tk.Button(self.color_mode_tab, text="Convert to CMYK", command=self.convert_to_cmyk)
        self.cmyk_button.grid(row=3, column=0, padx=10, pady=10, columnspan=2)

    def create_styletransfer_controls(self):
        # Advanced Deep Learning Features Button (Placeholder)
        self.deeplearning_button = tk.Button(self.advanced_tab, text="Open Style Transfer Feature", command=self.create_styletransfer_controls)
        self.deeplearning_button.grid(row=0, column=0, padx=10, pady=10)

    def create_intensity_controls(self):
        # Intensity Manipulation Buttons
        self.identity_button = tk.Button(self.intensity_tab, text="Apply Identity Transformation", command=self.apply_identity_transformation)
        self.identity_button.grid(row=0, column=0, padx=10, pady=10)

        self.negative_button = tk.Button(self.intensity_tab, text="Apply Negative Transformation", command=self.apply_negative_transformation)
        self.negative_button.grid(row=1, column=0, padx=10, pady=10)

        self.logarithmic_button = tk.Button(self.intensity_tab, text="Apply Logarithmic Transformation", command=self.apply_logarithmic_transformation)
        self.logarithmic_button.grid(row=2, column=0, padx=10, pady=10)

        self.powerlaw_button = tk.Button(self.intensity_tab, text="Apply Power-Law Transformation", command=self.apply_power_law_transformation)
        self.powerlaw_button.grid(row=3, column=0, padx=10, pady=10)

        self.inverse_logarithmic_button = tk.Button(self.intensity_tab, text="Apply Inverse Logarithmic Transformation", command=self.apply_inverse_logarithmic_transformation)
        self.inverse_logarithmic_button.grid(row=4, column=0, padx=10, pady=10)

        self.nth_power_button = tk.Button(self.intensity_tab, text="Apply nth Power Transformation", command=self.apply_nth_power_transformation)
        self.nth_power_button.grid(row=5, column=0, padx=10, pady=10)

        self.nth_root_button = tk.Button(self.intensity_tab, text="Apply nth Root Transformation", command=self.apply_nth_root_transformation)
        self.nth_root_button.grid(row=6, column=0, padx=10, pady=10)

    def create_segmentation_controls(self):
        # Segmentation Controls
        tk.Label(self.segmentation_tab, text="Segmentation Method:").grid(row=0, column=0, padx=10, pady=10)

        self.segmentation_method = tk.StringVar(value="Thresholding")

        tk.Radiobutton(self.segmentation_tab, text="Thresholding", variable=self.segmentation_method, value="Thresholding").grid(row=1, column=0, padx=10, pady=5)
        tk.Radiobutton(self.segmentation_tab, text="Edge Detection", variable=self.segmentation_method, value="Edge Detection").grid(row=2, column=0, padx=10, pady=5)
        tk.Radiobutton(self.segmentation_tab, text="Region-Based", variable=self.segmentation_method, value="Region-Based").grid(row=3, column=0, padx=10, pady=5)
        tk.Radiobutton(self.segmentation_tab, text="Clustering-Based", variable=self.segmentation_method, value="Clustering-Based").grid(row=4, column=0, padx=10, pady=5)

        self.apply_segmentation_button = tk.Button(self.segmentation_tab, text="Apply Segmentation", command=self.apply_segmentation)
        self.apply_segmentation_button.grid(row=5, column=0, padx=10, pady=10)

    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")]  # Add any other formats you want to support
        )
        
        if file_path:
            try:
                self.cv_image_path = file_path
                self.original_image = Image.open(file_path)
                self.modified_image = self.original_image.copy()
                self.display_image(self.original_image, self.original_image_label)  # Ensure original_image_label is defined
            except Exception as e:
                print(f"Error loading image: {e}")  # Handle error gracefully

    def save_image(self):
        if self.modified_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                self.modified_image.save(file_path)

    def reset_image(self):
        if self.cv_image_path:
            self.original_image = Image.open(self.cv_image_path)
            self.modified_image = self.original_image.copy()
            self.display_image(self.original_image, self.original_image_label)
            self.display_image(self.modified_image, self.modified_image_label)

    def display_image(self, image, label):
        img = ImageTk.PhotoImage(image)
        label.config(image=img)
        label.image = img

    def apply_filters(self):
        if self.modified_image:
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            if self.sharpening_var.get():
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                image_cv = cv2.filter2D(image_cv, -1, kernel)
            if self.smoothing_var.get():
                ksize = int(self.smoothing_scale.get())
                image_cv = cv2.GaussianBlur(image_cv, (ksize, ksize), 0)
            if self.embossing_var.get():
                kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 1]])
                image_cv = cv2.filter2D(image_cv, -1, kernel)
            if self.edge_detection_var.get():
                image_cv = cv2.Canny(image_cv, 100, 200)

            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.modified_image, self.modified_image_label)

    def apply_custom_rotation(self):
        if self.modified_image:
            angle = self.custom_rotation_scale.get()
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            (h, w) = image_cv.shape[:2]
            center = (w / 2, h / 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image_cv = cv2.warpAffine(image_cv, matrix, (w, h))
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.modified_image, self.modified_image_label)

    def rotate_fixed(self, angle):
        if self.modified_image:
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            (h, w) = image_cv.shape[:2]
            center = (w / 2, h / 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image_cv = cv2.warpAffine(image_cv, matrix, (w, h))
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.modified_image, self.modified_image_label)

    def crop_image(self):
        if self.modified_image:
            width = self.crop_width_var.get()
            height = self.crop_height_var.get()
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            (h, w) = image_cv.shape[:2]
            x_center, y_center = w // 2, h // 2
            x1 = max(x_center - width // 2, 0)
            y1 = max(y_center - height // 2, 0)
            x2 = min(x_center + width // 2, w)
            y2 = min(y_center + height // 2, h)
            image_cv = image_cv[y1:y2, x1:x2]
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.modified_image, self.modified_image_label)

    def flip_colors(self):
        if self.modified_image:
            self.modified_image = ImageOps.invert(self.modified_image)
            self.display_image(self.modified_image, self.modified_image_label)

    def convert_to_grayscale(self):
        if self.modified_image:
            self.modified_image = ImageOps.grayscale(self.modified_image)
            self.display_image(self.modified_image, self.modified_image_label)

    def convert_to_rgb(self):
        if self.modified_image:
            self.modified_image = self.modified_image.convert("RGB")
            self.display_image(self.modified_image, self.modified_image_label)

    def convert_to_hsv(self):
        if self.modified_image:
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_HSV2RGB))
            self.display_image(self.modified_image, self.modified_image_label)

    def convert_to_cmyk(self):
        if self.modified_image:
            self.modififed_image = self.modified_image.convert("CMYK")
            self.display_image(self.modified_image, self.modified_image_label)

    def apply_identity_transformation(self):
        if self.modified_image:
            self.modified_image = self.original_image.copy()
            self.display_image(self.modified_image, self.modified_image_label)

    def apply_negative_transformation(self):
        if self.modified_image:
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            image_cv = cv2.bitwise_not(image_cv)
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.modified_image, self.modified_image_label)

    def apply_logarithmic_transformation(self):
        if self.modified_image:
            # Convert the image to OpenCV format (BGR)
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            
            # Convert to float32 for logarithmic transformation
            image_cv = image_cv.astype(np.float32)
            
            # Apply logarithmic transformation
            image_cv = np.log1p(image_cv)
            
            # Normalize to the range [0, 255]
            image_cv = cv2.normalize(image_cv, None, 0, 255, cv2.NORM_MINMAX)
            
            # Convert back to uint8
            image_cv = np.uint8(image_cv)
            
            # Convert back to RGB format for display
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            
            # Display the modified image
            self.display_image(self.modified_image, self.modified_image_label)


    def apply_power_law_transformation(self):
        if self.modified_image:
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            gamma = 1.5  # Example gamma value
            image_cv = np.power(image_cv / 255.0, gamma) * 255
            image_cv = np.uint8(image_cv)
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.modified_image, self.modified_image_label)
    
    def apply_inverse_logarithmic_transformation(self):
         if self.modified_image:
            # Convert the image to OpenCV format (BGR)
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            
            # Convert to float32 for inverse logarithmic transformation
            image_cv = image_cv.astype(np.float32)
            
            # Apply inverse logarithmic transformation
            image_cv = np.expm1(image_cv)  # exp(x) - 1
            
            # Normalize to the range [0, 255]
            min_val = np.min(image_cv)
            max_val = np.max(image_cv)
            
            # Debugging statements
            print(f"Before normalization: min = {min_val}, max = {max_val}")
            
            if max_val > 0:  # Check to prevent division by zero
                image_cv = cv2.normalize(image_cv, None, 0, 255, cv2.NORM_MINMAX)
            else:
                print("Warning: Max value is zero, normalization skipped.")
            
            # Convert back to uint8
            image_cv = np.clip(image_cv, 0, 255)  # Ensure values are within [0, 255]
            image_cv = np.uint8(image_cv)
            
            # Convert back to RGB format for display
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            
            # Display the modified image
            self.display_image(self.modified_image, self.modified_image_label)


    def apply_nth_power_transformation(self):
        if self.modified_image:
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            n = 2  # Example nth power value
            image_cv = np.power(image_cv / 255.0, n) * 255
            image_cv = np.uint8(image_cv)
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.modified_image, self.modified_image_label)

    def apply_nth_root_transformation(self):
        if self.modified_image:
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            n = 2  # Example nth root value
            image_cv = np.power(image_cv / 255.0, 1/n) * 255
            image_cv = np.uint8(image_cv)
            self.modified_image = Image.fromarray(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
            self.display_image(self.modified_image, self.modified_image_label)

    def apply_segmentation(self):
        if self.modified_image:
            image_cv = cv2.cvtColor(np.array(self.modified_image), cv2.COLOR_RGB2BGR)
            method = self.segmentation_method.get()

            if method == "Thresholding":
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                _, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                self.modified_image = Image.fromarray(segmented)
            elif method == "Edge Detection":
                segmented = cv2.Canny(image_cv, 100, 200)
                self.modified_image = Image.fromarray(segmented)
            elif method == "Region-Based":
                gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
                _, labels = cv2.connectedComponents(gray)
                self.modified_image = Image.fromarray(labels.astype(np.uint8) * 255)
            elif method == "Clustering-Based":
                pixel_values = image_cv.reshape((-1, 3))
                pixel_values = np.float32(pixel_values)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
                k = 3
                _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                segmented_image = centers[labels.flatten()].reshape(image_cv.shape).astype(np.uint8)
                self.modified_image = Image.fromarray(segmented_image)

            self.display_image(self.modified_image, self.modified_image_label)

    def apply_deep_learning_features(self):
        if self.modified_image:
            # Convert the modified image to a NumPy array
            image_np = np.array(self.modified_image)

            # Apply Image Enhancement
            enhanced_image = self.enhance_image(image_np)
            self.modified_image = Image.fromarray(enhanced_image)
            self.display_image(self.modified_image)

            # Apply Style Transfer
            style_image_path = filedialog.askopenfilename(title="Select Style Image")  # Allow user to select style image
            if style_image_path:
                style_image = self.load_and_preprocess_image(style_image_path)
                self.modified_image = self.style_transfer(image_np, style_image)
                self.display_image(Image.fromarray(self.modified_image))

            # Apply GAN Image Generation
            generated_image = self.generate_image_with_gan()
            self.modified_image = Image.fromarray(generated_image)
            self.display_image(self.modified_image)

    def enhance_image(self, image):
        # Noise reduction using Gaussian blur
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Sharpening using an unsharp mask
        sharpened = cv2.addWeighted(image, 1.5, denoised, -0.5, 0)
        
        return sharpened

    def load_and_preprocess_image(self, path):
        img = load_img(path, target_size=(224, 224))  # type: ignore # Load and resize the image
        img = pil_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def style_transfer(self, content_image, style_image):
        # Load VGG19 model for feature extraction
        model = VGG19(weights='imagenet', include_top=False) # type: ignore

        # Define the layers to extract content and style features
        content_layer = 'block5_conv2'
        style_layers = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'
        ]

        # Create models for content and style
        content_model = model(inputs=model.input, outputs=model.get_layer(content_layer).output)
        style_models = [model(inputs=model.input, outputs=model.get_layer(layer).output) for layer in style_layers]

        # Extract features
        content_features = content_model(content_image)
        style_features = [style_model(style_image) for style_model in style_models]

        # Initialize a target image for optimization
        target_image = tf.Variable(content_image)

        # Optimization loop
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        for _ in range(100):  # Number of iterations
            with tf.GradientTape() as tape:
                target_content = content_model(target_image)
                target_styles = [style_model(target_image) for style_model in style_models]

                # Calculate losses
                content_loss = tf.reduce_mean((target_content - content_features) ** 2)
                style_loss = sum(tf.reduce_mean((self.gram_matrix(target_style) - self.gram_matrix(style)) ** 2)
                                 for target_style, style in zip(target_styles, style_features))
                
                total_loss = content_loss + style_loss
            grads = tape.gradient(total_loss, target_image)
            optimizer.apply_gradients([(grads, target_image)])

        return np.array(target_image.numpy()[0], dtype=np.uint8)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
