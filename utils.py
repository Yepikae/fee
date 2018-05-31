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

import sys
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


def print_processing(completion):
    """Print a loading bar."""
    percent = int(completion * 100)
    s = ''
    if percent < 10:
        s += '  '
    elif percent < 100:
        s += ' '
    s += ' '+str(percent)+'% ['
    for i in range(0, 60):
        if i <= int(completion * 60):
            s += '#'
        else:
            s += ' '
    s += ']'
    sys.stdout.write('\r')
    sys.stdout.flush()
    print(s, sep='', end='', flush=True)
