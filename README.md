# VirtualCam Filters with Face Detection and Image Overlay

This Python project captures webcam images and applies a series of basic image processing operations and filters using OpenCV. 
The modified video feed is streamed via a virtual camera using `pyvirtualcam`.

The project focuses on experimenting with image filters and statistical functions, and includes a special feature: face detection with image overlay!
The implemented image filters include Blur, Sharpen, Sobel, and Gabor filters.



## Installation guide
1. Create a virtual environment
```bash
python -m venv venv
```

2. Activate the virtual environment
```bash
venv\Scripts\activate       # on Windows
```

3. Install dependencies/packages
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
python run.py
```

#### Note:
- This project was tested with **Python 3.10**.
- `pyvirtualcam` currently does not support macOS 14 Ventura ([Source](https://github.com/letmaik/pyvirtualcam/issues/111))