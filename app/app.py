import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageEnhance, ImageOps, ImageFilter
import cv2
import numpy as np

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clean-View Image Processing App")

        # Initialize variables
        self.original_image = None
        self.modified_image = None

        # Create UI elements
        self.create_widgets()

    def create_widgets(self):
        # Open Image Button
        self.open_button = tk.Button(self.root, text="Open Image", command=self.open_image)
        self.open_button.grid(row=0, column=0, padx=10, pady=10)

        # Color Effect Options
        self.color_effects = ["Reset", "More Natural", "Black and white", "Vintage", "Cartoon",
                              "Vibrance", "Red", "Green", "Blue", "Gothic"]
        self.options_var = tk.StringVar()
        self.options_var.set("Options")
        self.options_menu = tk.OptionMenu(self.root, self.options_var, *self.color_effects, command=self.adjust_colors)
        self.options_menu.grid(row=0, column=1, padx=10, pady=10)

        # Filters and Adjustments
        self.create_filter_controls()

        # Image Labels
        self.original_image_label = tk.Label(self.root, text="Original Image")
        self.original_image_label.grid(row=1, column=0, padx=10, pady=10)
        self.modified_image_label = tk.Label(self.root, text="Modified Image")
        self.modified_image_label.grid(row=1, column=1, padx=10, pady=10)

    def create_filter_controls(self):
        # Filters Button
        self.filters_button = tk.Button(self.root, text="Apply Filters", command=self.apply_filters)
        self.filters_button.grid(row=2, column=0, padx=10, pady=10)

        # Filter Checkbuttons and Sliders
        self.hue_saturation_var = tk.BooleanVar()
        self.sharpening_var = tk.BooleanVar()
        self.gaussian_blur_var = tk.BooleanVar()
        self.edge_detection_var = tk.BooleanVar()

        tk.Checkbutton(self.root, text="Hue/Saturation", variable=self.hue_saturation_var).grid(row=3, column=0, padx=10, pady=5)
        self.hue_saturation_scale = tk.Scale(self.root, from_=1, to=10, orient="horizontal", length=200)
        self.hue_saturation_scale.grid(row=4, column=0, padx=10, pady=5)

        tk.Checkbutton(self.root, text="Sharpening", variable=self.sharpening_var).grid(row=3, column=1, padx=10, pady=5)
        self.sharpening_scale = tk.Scale(self.root, from_=1, to=10, orient="horizontal", length=200)
        self.sharpening_scale.grid(row=4, column=1, padx=10, pady=5)

        tk.Checkbutton(self.root, text="Gaussian Blur", variable=self.gaussian_blur_var).grid(row=3, column=2, padx=10, pady=5)
        self.gaussian_blur_scale = tk.Scale(self.root, from_=1, to=10, orient="horizontal", length=200)
        self.gaussian_blur_scale.grid(row=4, column=2, padx=10, pady=5)

        tk.Checkbutton(self.root, text="Edge Detection", variable=self.edge_detection_var).grid(row=5, column=0, padx=10, pady=5)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif *.bmp")])
        if file_path:
            self.original_image = Image.open(file_path)
            self.modified_image = self.original_image
            self.display_image(self.original_image, self.original_image_label)

    def adjust_colors(self, effect):
        if self.original_image:
            if effect == "Reset":
                self.modified_image = self.original_image
            elif effect == "More Natural":
                self.modified_image = ImageEnhance.Color(self.original_image).enhance(1.5)
            elif effect == "Black and white":
                self.modified_image = self.original_image.convert("L").convert("RGB")
            elif effect == "Vintage":
                self.modified_image = self.original_image.convert("RGB")
                r, g, b = self.modified_image.split()
                r = r.point(lambda i: i * 0.4)
                g = g.point(lambda i: i * 0.4)
                b = b.point(lambda i: i * 0.3)
                self.modified_image = Image.merge("RGB", (r, g, b))
            elif effect == "Cartoon":
                self.modified_image = self.original_image.convert("RGB")
                self.modified_image = ImageEnhance.Contrast(self.modified_image).enhance(7.0)
                self.modified_image = ImageEnhance.Color(self.modified_image).enhance(0)
            elif effect == "Vibrance":
                self.modified_image = self.original_image.convert("RGB")
                self.modified_image = ImageEnhance.Contrast(self.modified_image).enhance(2.0)
                self.modified_image = ImageEnhance.Color(self.modified_image).enhance(2.0)
            elif effect == "Red":
                self.modified_image = self.original_image.convert("RGB")
                r, g, b = self.modified_image.split()
                r = r.point(lambda i: i * 2.0)
                self.modified_image = Image.merge("RGB", (r, g, b))
            elif effect == "Green":
                self.modified_image = self.original_image.convert("RGB")
                r, g, b = self.modified_image.split()
                g = g.point(lambda i: i * 2.0)
                self.modified_image = Image.merge("RGB", (r, g, b))
            elif effect == "Blue":
                self.modified_image = self.original_image.convert("RGB")
                r, g, b = self.modified_image.split()
                b = b.point(lambda i: i * 3.0)
                self.modified_image = Image.merge("RGB", (r, g, b))
            elif effect == "Gothic":
                self.modified_image = self.original_image.convert("RGB")
                r, g, b = self.modified_image.split()
                r = r.point(lambda i: i * 0.3)
                g = g.point(lambda i: i * 0.3)
                b = b.point(lambda i: i * 0.3)
                self.modified_image = Image.merge("RGB", (r, g, b))
                self.modified_image = ImageEnhance.Brightness(self.modified_image).enhance(1.1)
                self.modified_image = ImageEnhance.Contrast(self.modified_image).enhance(3.0)
                self.modified_image = ImageEnhance.Color(self.modified_image).enhance(0)
            
            self.display_image(self.modified_image, self.modified_image_label)

    def apply_filters(self):
        if self.original_image:
            image = np.array(self.original_image)
            if self.hue_saturation_var.get():
                value = self.hue_saturation_scale.get() / 10.0
                hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                hsv_image[:, :, 1] = hsv_image[:, :, 1] * value
                image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
            if self.sharpening_var.get():
                value = self.sharpening_scale.get()
                kernel = np.array([[-1, -1, -1],
                                   [-1, value, -1],
                                   [-1, -1, -1]])
                image = cv2.filter2D(image, -1, kernel)
            if self.gaussian_blur_var.get():
                value = self.gaussian_blur_scale.get()
                kernel_size = int(2 * value + 1)
                image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            if self.edge_detection_var.get():
                gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray_image, 100, 200)
                image = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            self.modified_image = Image.fromarray(image)
            self.display_image(self.modified_image, self.modified_image_label)

    def display_image(self, image, label):
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
