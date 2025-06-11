import numpy as np
import math
from numba import njit

'''
Jit compiled function to increase performance.
Use some loops insteads of purely numpy functions.
If you face some compile errors using @njit, see: https://numba.pydata.org/numba-doc/dev/reference/numpysupported.html
In case you dont need performance boosts, remove the njit flag above the function
Do not use cv2 functions together with @njit
'''

@njit
def mean_numba(np_img):
    total = 0.0
    count = 0
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            for c in range(np_img.shape[2]):
                total += np_img[i, j, c]
                count += 1
    return total / count

@njit
def mode_numba(np_img):
    # Mode is the most frequent pixel value over all channels.
    # Assuming image pixels are uint8 (0-255)
    hist = np.zeros(256, dtype=np.int64)
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            for c in range(np_img.shape[2]):
                hist[np_img[i,j,c]] += 1
    max_count = 0
    mode_val = 0
    for val in range(256):
        if hist[val] > max_count:
            max_count = hist[val]
            mode_val = val
    return mode_val

@njit
def stddev_numba(np_img):
    m = mean_numba(np_img)
    total = 0.0
    count = 0
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            for c in range(np_img.shape[2]):
                diff = np_img[i, j, c] - m
                total += diff * diff
                count += 1
    return np.sqrt(total / count)

@njit
def max_numba(np_img):
    max_val = 0
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            for c in range(np_img.shape[2]):
                if np_img[i,j,c] > max_val:
                    max_val = np_img[i,j,c]
    return max_val

@njit
def min_numba(np_img):
    min_val = 255
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            for c in range(np_img.shape[2]):
                if np_img[i,j,c] < min_val:
                    min_val = np_img[i,j,c]
    return min_val

@njit
def linear_transform_numba(np_img, a, b):
    # Linear transform: output = a * input + b (clipped 0-255)
    out = np.empty_like(np_img)
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            for c in range(np_img.shape[2]):
                val = a * np_img[i,j,c] + b
                if val < 0:
                    val = 0
                elif val > 255:
                    val = 255
                out[i,j,c] = val
    return out

@njit
def entropy_numba(np_img):
    # Calculate entropy over all pixels and channels
    hist = np.zeros(256, dtype=np.int64)
    total_pixels = np_img.shape[0] * np_img.shape[1] * np_img.shape[2]
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            for c in range(np_img.shape[2]):
                hist[np_img[i,j,c]] += 1
    entropy = 0.0
    for i in range(256):
        p = hist[i] / total_pixels
        if p > 0:
            entropy -= p * np.log2(p)
    return entropy

@njit
def histogram_figure_numba(np_img):
    # Returns histograms for each channel as 256-bin arrays
    r_hist = np.zeros(256, dtype=np.int64)
    g_hist = np.zeros(256, dtype=np.int64)
    b_hist = np.zeros(256, dtype=np.int64)
    for i in range(np_img.shape[0]):
        for j in range(np_img.shape[1]):
            r_hist[np_img[i,j,0]] += 1
            g_hist[np_img[i,j,1]] += 1
            b_hist[np_img[i,j,2]] += 1
    return r_hist, g_hist, b_hist

@njit
def histogram_equalization_numba(np_img):
    # Apply histogram equalization channel-wise
    out = np.empty_like(np_img)
    for c in range(3):
        hist = np.zeros(256, dtype=np.int64)
        for i in range(np_img.shape[0]):
            for j in range(np_img.shape[1]):
                hist[np_img[i,j,c]] += 1

        cdf = np.zeros(256, dtype=np.float64)
        cdf[0] = hist[0]
        for i in range(1, 256):
            cdf[i] = cdf[i-1] + hist[i]
        cdf_min = 0
        for i in range(256):
            if cdf[i] != 0:
                cdf_min = cdf[i]
                break
        num_pixels = np_img.shape[0] * np_img.shape[1]

        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            lut[i] = np.uint8((cdf[i] - cdf_min) / (num_pixels - cdf_min) * 255)

        for i in range(np_img.shape[0]):
            for j in range(np_img.shape[1]):
                out[i,j,c] = lut[np_img[i,j,c]]

    return out


"""
Filters:
- Identity 
- Edge detection
- Blur filter
- Sharpen
- Sobel
- Gabor
"""
@njit
def identity_filter_numba(np_img):
    """Identity filter"""
    height, width, channels = np_img.shape
    out = np.zeros((height, width, channels), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            for c in range(channels):
                out[i,j,c] = np_img[i,j,c]
    return out

@njit
def blur_filter_numba(np_img, kernel_size=5):
    """Blur filter"""
    height, width, channels = np_img.shape
    pad = kernel_size // 2
    out = np.zeros_like(np_img)
    
    for c in range(channels):
        for i in range(pad, height - pad):
            for j in range(pad, width - pad):
                s = 0
                for ki in range(-pad, pad + 1):
                    for kj in range(-pad, pad + 1):
                        s += np_img[i + ki, j + kj, c]
                out[i, j, c] = s // (kernel_size * kernel_size)
    return out

@njit
def sharpen_filter_numba(np_img):
    """Sharpen filter"""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.int32)
    height, width, channels = np_img.shape
    out = np.zeros_like(np_img)
    
    for c in range(channels):
        for i in range(1, height-1):
            for j in range(1, width-1):
                val = 0
                for ki in range(-1, 2):
                    for kj in range(-1, 2):
                        val += kernel[ki+1, kj+1] * np_img[i+ki, j+kj, c]
                if val < 0:
                    val = 0
                elif val > 255:
                    val = 255
                out[i, j, c] = val
    return out

@njit
def gabor_filter_numba(np_img, ksize=7, sigma=2.0, theta=0.0, lambd=10.0, gamma=0.5, psi=0.0):
    """Gabor filter"""
    height, width, channels = np_img.shape
    half = ksize // 2
    out = np.zeros_like(np_img)

    # Precompute kernel
    kernel = np.zeros((ksize, ksize), dtype=np.float64)
    sigma_x = sigma
    sigma_y = sigma / gamma
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    for y in range(-half, half + 1):
        for x in range(-half, half + 1):
            x_theta = x * cos_theta + y * sin_theta
            y_theta = -x * sin_theta + y * cos_theta
            exp_term = math.exp(-0.5 * ((x_theta ** 2) / (sigma_x ** 2) + (y_theta ** 2) / (sigma_y ** 2)))
            cos_term = math.cos(2 * math.pi * x_theta / lambd + psi)
            kernel[y + half, x + half] = exp_term * cos_term

    # Normalize kernel
    kernel_sum = 0.0
    for i in range(ksize):
        for j in range(ksize):
            kernel_sum += kernel[i, j]
    if kernel_sum != 0:
        for i in range(ksize):
            for j in range(ksize):
                kernel[i, j] /= kernel_sum

    # Apply convolution
    for c in range(channels):
        for i in range(half, height - half):
            for j in range(half, width - half):
                val = 0.0
                for ki in range(-half, half + 1):
                    for kj in range(-half, half + 1):
                        val += kernel[ki + half, kj + half] * np_img[i + ki, j + kj, c]
                # Clip to valid range
                if val < 0:
                    val = 0
                elif val > 255:
                    val = 255
                out[i, j, c] = np.uint8(val)

    return out

@njit
def sobel_filter_numba(np_img):
    """Sobel filter"""
    # Simple edge detection using Sobel filter on grayscale (convert RGB to grayscale first)
    height, width = np_img.shape[0], np_img.shape[1]
    out = np.zeros((height, width), dtype=np.uint8)

    # Sobel kernels
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.int32)
    Gy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]], dtype=np.int32)

    # Convert to grayscale luminance (simple average)
    gray = np.empty((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            gray[i,j] = (np_img[i,j,0] + np_img[i,j,1] + np_img[i,j,2]) // 3

    # Apply Sobel filter (skip edges)
    for i in range(1, height-1):
        for j in range(1, width-1):
            sx = 0
            sy = 0
            for ki in range(-1, 2):
                for kj in range(-1, 2):
                    sx += Gx[ki+1, kj+1] * gray[i+ki, j+kj]
                    sy += Gy[ki+1, kj+1] * gray[i+ki, j+kj]
            mag = np.sqrt(sx*sx + sy*sy)
            if mag > 255:
                mag = 255
            out[i,j] = np.uint8(mag)

    return out