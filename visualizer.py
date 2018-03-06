"""Todo."""

import numpy as np
import cv2
from memory_profiler import profile

def get_minimal_image(points, width, height, weight=10, color=(0, 255, 0)):
    """Create a minimal image of the landmark points."""
    # Create a black image.
    img = np.zeros((height, width, 3), np.uint8)
    # Draw a circle for each landmark point.
    idx = 0
    while idx < len(points):
        cv2.circle(img, (points[idx], points[idx+1]), weight,
                   color, -1)
        idx = idx + 2
    return img

# @profile(precision=8)
def get_one_color_image(points, width, height, weight=10):
    """Create a minimal image of the landmark points."""
    # Create a black image.
    img = np.zeros((height, width, 1), np.uint8)
    # Draw a circle for each landmark point.
    idx = 0
    while idx < len(points):
        cv2.circle(img, (points[idx], points[idx+1]), weight,
                   255, -1)
        idx = idx + 2
    return img

def get_one_color_pix_image(points, width, height):
    """Create a minimal image of the landmark points."""
    # Create a black image.
    img = np.zeros((height, width, 1), np.uint8)
    # Draw a circle for each landmark point.
    idx = 0
    while idx < len(points):
        if points[idx] < width and points[idx] > 0:
            if points[idx+1] > 0 and points[idx+1] < height:
                img[points[idx + 1]][points[idx]] = 255
        idx = idx + 2
    return img

def get_overlay_image(points, source, weight=10):
    """Return an overlay with the landmark points over the source image."""
    idx = 0
    while idx < len(points):
        cv2.circle(source, (points[idx], points[idx+1]), weight,
                   (0, 255, 0), -1)
        idx = idx + 2
    return source
