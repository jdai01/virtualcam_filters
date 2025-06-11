import keyboard
import numpy as np
import cv2
import time
import os
from capturing import VirtualCamera
from overlays import initialize_hist_figure, plot_overlay_to_image, plot_strings_to_image, update_histogram
from basics import *
from special import *

def custom_processing(img_source_generator, processing_function):
    # Initialize histogram figure and plotting objects
    fig, ax, background, plot_r, plot_g, plot_b = initialize_hist_figure()
    
    for input_data in img_source_generator:
        # Apply processing function (e.g., edge detection)
        processed_output = processing_function(input_data.copy())
         
        # Convert single-channel (e.g., grayscale) to 3-channel format
        processed_output_colored = np.stack((processed_output,) * 3, axis=-1)
        
        # Generate color histograms from original input
        hist_r, hist_g, hist_b = histogram_figure_numba(input_data)
        update_histogram(
            fig, ax, background,
            plot_r, plot_g, plot_b, hist_r, hist_g, hist_b
        )
        
        # Overlay the visualization on the processed image
        processed_output_colored = plot_overlay_to_image(processed_output_colored, fig)
        
        # Add annotation text to the image
        annotation_lines = ["Processed Output", "Press 'h' to print"]
        processed_output_colored = plot_strings_to_image(processed_output_colored, annotation_lines)
        
        # Handle key press interaction
        if keyboard.is_pressed('h'):
            # Ensure 'img' directory exists
            folder = 'img'
            os.makedirs(folder, exist_ok=True)

            # Create a unique filename using timestamp
            filename = f"{folder}/output_{int(time.time())}.png"
            cv2.imwrite(filename, processed_output_colored)
            print(f"Image saved as {filename}")
        
        # Yield original and processed images
        yield input_data, processed_output_colored



def main():
    # change according to your settings
    width, height, fps = 1280, 720, 30
    vc = VirtualCamera(fps, width, height)

    # Initialize face detector and load emoji globally once
    global face_cascade, cat_emoji
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cat_emoji = cv2.imread('cat_emoji.png', cv2.IMREAD_UNCHANGED)
    
    vc.virtual_cam_interaction(
        custom_processing(
            # either camera stream
            vc.capture_cv_video(0, bgr_to_rgb=True),
            
            # or your window screen
            # vc.capture_screen(),

            # Processing function
            # processing_function=identity_filter_numba
            # processing_function=blur_filter_numba
            # processing_function=sharpen_filter_numba
            # processing_function=gabor_filter_numba
            processing_function=sobel_filter_numba
            # processing_function=processing_with_cat_overlay
        ), 
        preview=True
    )

if __name__ == "__main__":
    main()