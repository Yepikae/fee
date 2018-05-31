"""Todo."""

import dlib
import os
from enum import Enum
import math
import fee.visualizer as fviz
import fee.utils as fu


class Cat(Enum):
    """Landmarks points categories."""

    ALL = 1
    JAW = 2
    RIGHT_EYE = 3
    LEFT_EYE = 4
    EYES = 5
    RIGHT_EYEBROW = 6
    LEFT_EYEBROW = 7
    EYEBROWS = 8
    NOSE = 9
    MOUTH = 10
    MOUTH_CROSS = 11


class FLandmarks:
    """
    Facial landmarks main class.

    Stores a set of points, the facial landmarks points.
    """

    def __init__(self):
        """Constructor."""
        self.__points__ = []
        self.__centroid__ = []
        self.__bounds__ = None

    def extract_points(self, image, predictor, predictorpath=""):
        """Extract the landmarks points for the first face detected.

        Return an array of shapes (see dlib) aka the landmarks points.
        """
        if predictor is None:
            if predictorpath == "":
                filepath = os.path.realpath(__file__)
                predictorpath = filepath[:-12] + "sp68fl.dat"
            predictor = dlib.shape_predictor(predictorpath)
        # Use a frontal face detector to retrieve the faces
        detector = dlib.get_frontal_face_detector()
        faces = detector(image, 0)
        if len(faces) != 0:
            shape = predictor(image, faces[0])
            for idx in range(0, 68):
                self.__points__.append(shape.part(idx).x)
                self.__points__.append(shape.part(idx).y)
            self.__calculate_bounds__()
            self.__calculate_centroid__()

    def set_points(self, points):
        """Copy points in to self.__points__."""
        self.__points__ = list(points)
        self.__calculate_bounds__()
        self.__calculate_centroid__()

    def get_l2bf(self):
        """Return an array of 'landmark to barycenter features'."""
        l2bf = []
        i = 0
        while i < len(self.__points__):
            x = self.__points__[i] - self.__centroid__[0]
            y = self.__points__[i+1] - self.__centroid__[1]
            magnitude = math.sqrt(x*x+y*y)
            l2bf.append(magnitude)
            angle = math.acos(x/magnitude)*180/math.pi
            l2bf.append(angle)
            i += 2
        return l2bf

    def get_normalized_l2bf(self):
        """Return a normalized array of 'landmark to barycenter features'."""
        l2bf = []
        i = 0
        maxmag = 0
        while i < len(self.__points__):
            x = self.__points__[i] - self.__centroid__[0]
            y = self.__points__[i+1] - self.__centroid__[1]
            magnitude = math.sqrt(x*x+y*y)
            if magnitude > maxmag:
                maxmag = magnitude
            l2bf.append(magnitude)
            angle = math.acos(x/magnitude)*180/math.pi / 360
            l2bf.append(angle)
            i += 2
        i = 0
        while i < len(l2bf):
            l2bf[i] /= maxmag
            i += 2
        return l2bf

    def get_frame(self, FW, FH, FC=True, OPP=False):
        """Return a minimal image built with the points."""
        pts = []
        # Center the points
        if FC:
            pts = fu.center_points(self.__points__,
                                   fu.get_bounds(self.__points__))
        else:
            pts = self.__points__
        # Scale the points
        if not OPP:
            pts = fu.rescale_points(pts,
                                    fu.get_bounds(pts),
                                    {"width": FW, "height": FH})
            return fviz.get_one_color_image(pts, FW, FH, 1)
        # Return the picture
        else:
            return fviz.get_one_color_pix_image(pts, FW, FH)

    def __calculate_bounds__(self):
        """Calculate the bounds of the __points__."""
        points = self.__points__
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
        self.__bounds__ = {"left": minX,
                           "width": maxX-minX,
                           "top": minY,
                           "height": maxY-minY}

    def __calculate_centroid__(self):
        """Calculate the centroid of the __points__."""
        points = self.__points__
        i = sumX = sumY = 0
        while i < len(points):
            sumX += points[i]
            sumY += points[i+1]
            i += 2
        self.__centroid__ = [2*sumX/len(points),
                             2*sumY/len(points)]

    def get_centered_points(self, bounds, padding=4):
        """Center the points according to the bounding box."""
        id = 0
        result = []
        points = self.__points__
        while id < len(points):
            # Position - Bounds + Frame Border
            result.append(points[id] - bounds["left"] + padding)
            result.append(points[id + 1] - bounds["top"] + padding)
            id = id + 2
        return result

    def get_rescaled_points(self, bounds, dest, padding=4):
        """Rescale the points from the bounds to the dest."""
        points = self.__points__
        result = []
        rx = dest["width"] / (bounds["width"]+padding*2)
        ry = dest["height"] / (bounds["height"]+padding*2)
        i = 0
        while i < len(points):
            result.append(math.floor(points[i] * rx))
            result.append(math.floor(points[i + 1] * ry))
            i = i + 2
        return result

    def get_bounds(self):
        """Return __bounds__."""
        return self.__bounds__

    def get_all_points(self):
        """Return __points__."""
        return self.__points__


def __get_points__(landmarks, min, max, normalized, flat):
    """Return the landmark points in the range [min,max[."""
    result = []
    # If we need normalized values
    if normalized:
        left = landmarks.rect.left()
        width = landmarks.rect.width()
        top = landmarks.rect.top()
        height = landmarks.rect.height()
    # For each point, fill the return list
    for idx in range(min, max):
        if flat:
            if normalized:
                result.append((landmarks.part(idx).x-left) / width)
                result.append((landmarks.part(idx).y-top) / height)
            else:
                result.append(landmarks.part(idx).x)
                result.append(landmarks.part(idx).y)
        else:
            result.append([landmarks.part(idx).x,
                           landmarks.part(idx).y])
    return result


def get_points(landmarks, cat, normalized=False, flat=True):
    """Return the landmarks points corresponding to a category."""
    if cat is Cat.ALL:
        return get_all_points(landmarks, normalized, flat)


def get_bounds(landmarks, cat=Cat.ALL):
    """Return the landmarks points bounds position."""
    if cat is Cat.ALL:
        return {
                "top": landmarks.rect.top(),
                "height": landmarks.rect.height(),
                "left": landmarks.rect.left(),
                "width": landmarks.rect.width()
            }


def get_all_points(landmarks, normalized=False, flat=True):
    """Return a copy of all the landmark points.

    Set flat to True to return an inline array [x0, y0, x1, y1]
    Set flat to False to return a 2d array [[x0,y0], [x1,y1]]
    """
    return __get_points__(landmarks, 0, 68, normalized, flat)


def get_mouth(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of the mouth."""
    return __get_points__(landmarks, 48, 68, normalized,)


def get_nose(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of the nose."""
    return __get_points__(landmarks, 27, 36, normalized,)


def get_left_eye(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of the left eye."""
    return __get_points__(landmarks, 42, 48, normalized, flat)


def get_right_eye(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of the right eye."""
    return __get_points__(landmarks, 36, 42, normalized, flat)


def get_eyes(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of both the eyes."""
    result = get_right_eye(landmarks, normalized, flat)
    for c, coords in enumerate(get_left_eye(landmarks, normalized, flat)):
        result.append(coords)
    return result


def get_left_eyebrow(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of the left eyebrow."""
    return __get_points__(landmarks, 22, 27, normalized, flat)


def get_right_eyebrow(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of the right eyebrow."""
    return __get_points__(landmarks, 17, 22, normalized, flat)


def get_eyebrows(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of both the eyebrows."""
    result = get_right_eyebrow(landmarks, normalized, flat)
    for c, coords in enumerate(get_left_eyebrow(landmarks, normalized, flat)):
        result.append(coords)
    return result


def get_jaw(landmarks, normalized=False, flat=True):
    """Return the landmark points of the edge of the jaw."""
    return __get_points__(landmarks, 0, 17, normalized,)


def get_specific_points(landmarks, indexes, normalized=False, flat=True):
    """Return the landmark points of the indexes passed in parameter."""
    result = []
    # If we need normalized values
    if normalized:
        left = landmarks.rect.left()
        width = landmarks.rect.width()
        top = landmarks.rect.top()
        height = landmarks.rect.height()
    # For each point, fill the return list
    for idx, index in enumerate(indexes):
        if flat:
            if normalized:
                result.append((landmarks.part(index).x-left) / width)
                result.append((landmarks.part(index).y-top) / height)
            else:
                result.append(landmarks.part(index).x)
                result.append(landmarks.part(index).y)
        else:
            result.append([landmarks.part(index).x,
                           landmarks.part(index).y])
    return result
