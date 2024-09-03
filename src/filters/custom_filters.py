from PIL import ImageFilter

def apply(image):
    # Example of a custom filter
    return image.filter(ImageFilter.CONTOUR)
