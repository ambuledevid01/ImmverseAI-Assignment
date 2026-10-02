# Image preprocessing and enhancement module for manuscript images.
import cv2
import numpy as np
from typing import Tuple, Optional
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)
def denoise_image(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
def binarize_image(gray_image: np.ndarray) -> np.ndarray:
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
def deskew_image(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) > 15.0:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (800, 800)) -> np.ndarray:
    if image is None:
        raise ValueError("Input image is None. Please check file path.")
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
def preprocess_image(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
    enable_enhancement: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise FileNotFoundError(f"Could not load image from relative path: {image_path}")
    processed_image = original_image.copy()
    if enable_enhancement:
        processed_image = deskew_image(processed_image)
        processed_image = apply_clahe(processed_image)
        processed_image = denoise_image(processed_image)
    if target_size is not None:
        processed_image = resize_image(processed_image, target_size)
    return original_image, processed_image
