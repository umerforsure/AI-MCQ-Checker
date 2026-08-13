"""
Preprocessing: grayscale, denoise, binarize (Otsu or adaptive).
"""

import cv2


def to_grayscale(image_bgr):
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def denoise(image_gray, method="auto"):
    if method == "median":
        return cv2.medianBlur(image_gray, 3)
    elif method == "nlmeans":
        return cv2.fastNlMeansDenoising(image_gray, h=10)
    else:
        return cv2.medianBlur(image_gray, 3)


def binarize(image_gray, method="otsu"):
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image_gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25,
            C=10,
        )
    else:
        _, binary = cv2.threshold(
            image_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
    return binary


def preprocess(image_bgr, denoise_method="auto", binarize_method="otsu"):
    gray = to_grayscale(image_bgr)
    denoised = denoise(gray, method=denoise_method)
    binary = binarize(denoised, method=binarize_method)
    return denoised, binary
