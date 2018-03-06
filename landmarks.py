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

    def set_points(self, points):
        """Copy points in to self.__points__."""
        self.__points__ = list(points)
        i = sumX = sumY = 0
        while i < len(points):
            sumX += points[i]
            sumY += points[i+1]
            i += 2
        self.__centroid__ = [2*sumX/len(points),
                             2*sumY/len(points)]
        self.__bounds__ = fu.get_bounds(self.__points__)

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

    # @profile(precision=8)
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


def get_landmark_points(image, predictor, predictorpath=""):
    """Extract the landmarks points for each faces.

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
    lm = []
    for k, d in enumerate(faces):
        shape = predictor(image, d)
        lm.append(shape)
    return lm


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
