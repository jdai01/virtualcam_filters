import cv2
import os
from basics import *

# === GLOBAL VARIABLES ===
# Face detection with OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load cat emoji with proper path
cat_emoji_path = os.path.join(os.path.dirname(__file__), 'cat_emoji.png')
cat_emoji = cv2.imread(cat_emoji_path, cv2.IMREAD_UNCHANGED)
if cat_emoji is None:
    raise FileNotFoundError(f"Could not load '{cat_emoji_path}'. Please check the path and file integrity.")

__all__ = ["face_cascade", "cat_emoji", "processing_with_cat_overlay"]

# === FUNCTION DEFINITIONS ===
def overlay_image_alpha(img, img_overlay, pos):
    """Overlay img_overlay on top of img at position pos and blend using alpha channel"""
    x, y = pos
    overlay_h, overlay_w = img_overlay.shape[:2]

    # Clip overlay if it goes out of bounds
    if x >= img.shape[1] or y >= img.shape[0]:
        return img
    w = min(overlay_w, img.shape[1] - x)
    h = min(overlay_h, img.shape[0] - y)

    if w <= 0 or h <= 0:
        return img

    overlay = img_overlay[0:h, 0:w]
    img_crop = img[y:y+h, x:x+w]

    if overlay.shape[2] < 4:
        # No alpha channel, just replace
        img[y:y+h, x:x+w] = overlay
        return img

    # Separate color and alpha channels
    overlay_img = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    # Blend overlay and image crop
    img_crop[:] = (1.0 - mask) * img_crop + mask * overlay_img

    return img

def replace_face_with_cat(img_bgr, face_cascade, cat_emoji):
    """
    Detect faces in the image and overlay a cat emoji over each detected face.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    
    for (x, y, w, h) in faces:
        # Resize cat emoji to face size
        cat_resized = cv2.resize(cat_emoji, (w, h), interpolation=cv2.INTER_AREA)
        # Overlay cat emoji on original image
        img_bgr = overlay_image_alpha(img_bgr, cat_resized, (x, y))
    
    return img_bgr

def processing_with_cat_overlay(img_rgb, processing_function=identity_filter_numba):
    """Cat overlay (Special)"""
    # Convert RGB to BGR for OpenCV face detection
    img_bgr = img_rgb[..., ::-1]

    # Detect face and overlay cat emoji
    img_bgr_with_cat = replace_face_with_cat(img_bgr, face_cascade, cat_emoji)

    # Convert back to RGB for further processing
    img_rgb_with_cat = img_bgr_with_cat[..., ::-1]

    # Apply identity on img_rgb_with_cat
    processed = processing_function(img_rgb_with_cat)

    return processed


