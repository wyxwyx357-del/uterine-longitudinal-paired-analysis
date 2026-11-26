import cv2
import numpy as np


def local_contrast(gray: np.ndarray, sigma: float = 5.) -> np.ndarray:
    source = gray.astype(np.float32)
    mean = cv2.GaussianBlur(source, (0, 0), sigma)
    square = cv2.GaussianBlur(source * source, (0, 0), sigma)
    std = np.sqrt(np.maximum(square - mean * mean, 0))
    mapped = 128 + 35 * (source - mean) / (std + 5)
    return np.uint8(np.where(gray > 12, np.clip(np.rint(mapped), 0, 255), 0))
