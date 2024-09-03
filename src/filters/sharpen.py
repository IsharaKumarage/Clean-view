from PIL import ImageFilter

def apply(image):
    return image.filter(ImageFilter.SHARPEN)
