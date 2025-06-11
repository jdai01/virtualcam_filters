# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:58:41 2021

@author: droes
"""

import pyvirtualcam
import numpy as np
import cv2 # conda install opencv
from PIL import ImageGrab # conda install pillow
from matplotlib import pyplot as plt # conda install matplotlib
import keyboard


class VirtualCamera:
    def __init__(self, fps, width, height):
        self.fps = fps
        self.width = width
        self.height = height
        
    def capture_screen(self, plt_inside=False, alt_width=0, alt_height=0):
        '''
        Represents the content of the primary monitor.
        Can be used to quickly test your application.
        '''
        width = alt_width if alt_width > 0 else self.width
        height = alt_height if alt_height > 0 else self.height
        while True:
            # grab is a slow method!
            img = ImageGrab.grab(bbox=(0, 0, width, height)) #x, y, w, h
            img_np = np.array(img)
            #img_np = np.zeros(shape=(height, width, 3), dtype=np.uint8)
            if plt_inside:
                plt.imshow(img_np)
                plt.axis('off')
                plt.show()
            yield img_np

            
    def capture_cv_video(self, camera_id, bgr_to_rgb=False):
        '''
        Establishes the connection to the camera via opencv
        Source: https://github.com/letmaik/pyvirtualcam/blob/master/samples/webcam_filter.py
        '''
        cv_vid = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

        if not cv_vid.isOpened():
            raise RuntimeError('Video-Output cannot be opened.')
            
        cv_vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cv_vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cv_vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cv_vid.set(cv2.CAP_PROP_FPS, self.fps)

        # Tatsächliche Einstellungen können sich von den oberhalb festgelegten dennoch unterscheiden!
        width = int(cv_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cv_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cv_vid.get(cv2.CAP_PROP_FPS)
        print(f'Camera properties: ({width}x{height} @ {fps_in}fps)')
        
        while True:
            ret, frame = cv_vid.read()
            if not ret:
                raise RuntimeError('Camera image cannot be loaded.')
            if bgr_to_rgb:
                frame = frame[...,::-1]
                
            if keyboard.is_pressed('q'):
                # quit camera stream
                cv_vid.release()
                return
                
            yield frame

    
    def virtual_cam_interaction(self, img_generator, print_fps=True, preview=True):
        '''
        Provides a virtual camera and optionally shows both original and processed video in a single OpenCV popup window.
        img_generator must represent a function that acts as a generator and returns image data.
        '''
        print('Quit camera stream with "q"')
        with pyvirtualcam.Camera(width=self.width, height=self.height, fps=self.fps, print_fps=print_fps) as cam:
            if preview:
                window_name = 'Original (Top) | Processed (Bottom)'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, int(self.width/2), self.height)  # width x height

            
            for orig_img, proc_img in img_generator:
                cam.send(proc_img) # provide the image
                cam.sleep_until_next_frame() # wait for next frame (fps dependent)

                if preview:
                    # Convert both images from RGB to BGR for OpenCV display
                    orig_bgr = orig_img[..., ::-1]
                    proc_bgr = proc_img[..., ::-1]
                    
                    # Concatenate horizontally
                    combined = np.vstack((orig_bgr, proc_bgr))
                    
                    # Show in one window
                    cv2.imshow(window_name, combined)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
            if preview:
                cv2.destroyAllWindows()
