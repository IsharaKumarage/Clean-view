import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import os
import tensorflow as tf
from tensorflow.keras.applications import vgg19 # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import numpy as np

# Initialize CustomTkinter with a white theme
ctk.set_appearance_mode("Light")  # Set to "Light" for a white theme
ctk.set_default_color_theme("blue")  # You can customize the color theme here

# Create the main window
root = ctk.CTk()
root.geometry("1200x800")  # Increased width for a better layout
root.title("Style Transfer App")

# Global variables for storing images
content_image = None
style_image = None

# Function to preprocess the image for VGG19
def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

# Function to deprocess the image to display it after style transfer
def deprocess_image(x):
    x = x.reshape((224, 224, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

# Function to upload content image
def upload_content_image():
    global content_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path and os.path.isfile(file_path):
        try:
            content_image = Image.open(file_path)
            img_resized = content_image.resize((400, 300), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(400, 300))
            content_image_label.configure(image=ctk_img)
            content_image_label.image = ctk_img
            content_image_label.text = ""
        except PermissionError as e:
            print(f"Permission denied: {e}")
    else:
        print("Invalid file path or file doesn't exist.")

# Function to upload style image
def upload_style_image():
    global style_image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
    if file_path and os.path.isfile(file_path):
        try:
            style_image = Image.open(file_path)
            img_resized = style_image.resize((400, 300), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(400, 300))
            style_image_label.configure(image=ctk_img)
            style_image_label.image = ctk_img
            style_image_label.text = ""
        except PermissionError as e:
            print(f"Permission denied: {e}")
    else:
        print("Invalid file path or file doesn't exist.")

# Function for style transfer
def style_transfer(content_path, style_path, iterations=100):
    content_image = preprocess_image(content_path)
    style_image = preprocess_image(style_path)

    vgg = vgg19.VGG19(weights="imagenet", include_top=False)
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

    content_model = tf.keras.Model([vgg.input], [vgg.get_layer(content_layer).output])
    style_model = tf.keras.Model([vgg.input], [vgg.get_layer(layer).output for layer in style_layers])

    content_features = content_model(content_image)
    style_features = style_model(style_image)

    generated_image = tf.Variable(content_image, dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5.0)

    for i in range(iterations):
        with tf.GradientTape() as tape:
            gen_content_features = content_model(generated_image)
            content_loss = tf.reduce_mean(tf.square(gen_content_features - content_features))

            gen_style_features = style_model(generated_image)
            style_loss = 0
            for gen_style, style_feature in zip(gen_style_features, style_features):
                style_loss += tf.reduce_mean(tf.square(gen_style - style_feature))

            total_loss = content_loss + style_loss

        grads = tape.gradient(total_loss, generated_image)
        opt.apply_gradients([(grads, generated_image)])

        if i % 10 == 0:
            print(f"Iteration {i}, Total Loss: {total_loss.numpy()}")

    output_img = deprocess_image(generated_image.numpy())
    return output_img

# Function to process style transfer and display the result
def process_style_transfer():
    if content_image and style_image:
        # Show processing indicator
        processing_label.configure(text="Processing... Please wait.")
        root.update()  # Update the UI
        
        processed_image = style_transfer(content_image.filename, style_image.filename)
        
        # Display the result
        img_resized = Image.fromarray(processed_image).resize((400, 300))
        ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(400, 300))
        result_image_label.configure(image=ctk_img)
        result_image_label.image = ctk_img
        
        processing_label.configure(text="Processing completed!")
    else:
        print("Please upload both content and style images before proceeding.")

# Function to save the result image
def save_result_image():
    if result_image_label.image:
        save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")])
        if save_path:
            result_image = Image.fromarray(np.array(result_image_label.image))
            result_image.save(save_path)
            print(f"Image saved at: {save_path}")
    else:
        print("No image to save.")

# UI Components
toolbar = ctk.CTkFrame(root, height=50)
toolbar.pack(side="top", fill="x")

# Toolbar Buttons
upload_content_button = ctk.CTkButton(toolbar, text="Upload Content Image", command=upload_content_image, width=15)
upload_content_button.pack(side="left", padx=5, pady=5)

upload_style_button = ctk.CTkButton(toolbar, text="Upload Style Image", command=upload_style_image, width=15)
upload_style_button.pack(side="left", padx=5, pady=5)

process_button = ctk.CTkButton(toolbar, text="Process Style Transfer", command=process_style_transfer, width=15)
process_button.pack(side="left", padx=5, pady=5)

save_button = ctk.CTkButton(toolbar, text="Save Result Image", command=save_result_image, width=15)
save_button.pack(side="left", padx=5, pady=5)

# Frame for images
image_frame = ctk.CTkFrame(root)
image_frame.pack(pady=20)

content_image_label = ctk.CTkLabel(image_frame, text="Content Image", width=400, height=300, fg_color="white", corner_radius=5)
content_image_label.grid(row=0, column=0, padx=20, pady=20)

style_image_label = ctk.CTkLabel(image_frame, text="Style Image", width=400, height=300, fg_color="white", corner_radius=5)
style_image_label.grid(row=0, column=1, padx=20, pady=20)

# Processing label
processing_label = ctk.CTkLabel(image_frame, text="", fg_color="white", corner_radius=5)
processing_label.grid(row=1, column=0, columnspan=2, padx=20, pady=10)

# Result label
result_image_label = ctk.CTkLabel(image_frame, text="Result Image", width=400, height=300, fg_color="white", corner_radius=5)
result_image_label.grid(row=2, column=0, columnspan=2, padx=20, pady=20)

# Start the Tkinter event loop
root.mainloop()
