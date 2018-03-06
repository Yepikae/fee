# TODO ?


# def vertical_align_landmark_points(image, topindex, bottomindex):
# """Return the landmarks with 2 points aligned vertically."""
# rows, cols, dim = image.shape
#
# points = sel.get_landmark_points([51, 57])
# print(points)
# vector_u = [0, -1]
# vector_v = [points[0]-points[2], points[1]-points[3]]
# cos_alpha = -vector_v[1] / math.sqrt(vector_v[0]*vector_v[0] +
#                                      vector_v[1]*vector_v[1])
# print(cos_alpha)
# print(math.acos(cos_alpha))
# print(math.acos(cos_alpha) * 180 / math.pi)
# angle = math.acos(cos_alpha) * 180 / math.pi
#
# if(vector_v[0] < 0):
#     angle = -angle
#
# M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
# dst = cv2.warpAffine(res, M, (cols, rows))

import math

def normalize_landmarks_points(points):
    """Normalise a list of points."""
    xmax, ymax, idx = 0
    result = []
    # Look for max x, y
    while idx < len(points):
        if xmax < points[idx]:
            xmax = points[idx]
        if ymax < points[idx + 1]:
            ymax = points[idx + 1]
        idx = idx + 2
    # Normalize
    idx = 0
    while idx < len(points):
        result.append(points[idx] / xmax)
        result.append(points[idx + 1] / ymax)
        idx = idx + 2
    return result


def get_bounds(points):
    """Return the bounds of a points cloud."""
    minX = maxX = points[0]
    minY = maxY = points[1]
    i = 2
    while i < len(points):
        if points[i] < minX:
            minX = points[i]
        elif points[i] > maxX:
            maxX = points[i]
        if points[i + 1] < minY:
            minY = points[i + 1]
        elif points[i + 1] > maxY:
            maxY = points[i + 1]
        i = i + 2
    return {"left": minX,
            "width": maxX-minX,
            "top": minY,
            "height": maxY-minY}


def center_points(points, bounds, padding=4):
    """Center the points according to the bounding box."""
    id = 0
    result = []
    while id < len(points):
        # Position - Bounds + Frame Border
        result.append(points[id] - bounds["left"] + padding)
        result.append(points[id + 1] - bounds["top"] + padding)
        id = id + 2
    return result


def rescale_points(src, bounds, dest, padding=4):
    """Rescale the points from the bounds to the dest."""
    points = []
    rx = dest["width"] / (bounds["width"]+padding*2)
    ry = dest["height"] / (bounds["height"]+padding*2)
    i = 0
    while i < len(src):
        points.append(math.floor(src[i] * rx))
        points.append(math.floor(src[i + 1] * ry))
        i = i + 2
    return points
