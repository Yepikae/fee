"""This module aims to read/write files related to the fee module.
This module helps generating the landmark points or features json files.
"""
import json
import fee.landmarks
import cv2
import numpy as np
import os
import glob
from fee.classification import Expression as Exp
from fee.data import FlandmarksDataStorage
from fee.data import FlandmarksData
from fee.landmarks import FLandmarks

# TODO : landmarks_json . Create a normalized enum.



# File creation ##############################################################
# ############################################################################


def create_dataset_json(filepath, dataset):
    """Create a json file with the dataset.

    Param: filepath, string. Path of the file to write.
    Param: dataset, json object. The dataset to write.
    """
    file = open(filepath, 'w')
    file.write(json.dumps(dataset))
    file.close()


# Read JSON Dataset Files ####################################################
# ############################################################################

def get_dataset_from_multiple_json(folder_path):
    datas = FlandmarksDataStorage()
    for id, f in enumerate(glob.glob(os.path.join(folder_path, "*.json"))):
        # Open the data file
        src = json.load(open(f, 'r'))["datas"]
        for j, src_elem in enumerate(src):
            data = FlandmarksData(src_elem["file"],
                                  Exp.from_str(src_elem["oclass"]))
            # Save the n frames points
            for pos in range(0, len(src_elem["values"])):
                if src_elem["values"][pos] is not None:
                    fl = FLandmarks()
                    fl.set_points(src_elem["values"][pos]["points"])
                    data.add_flandmarks(fl)
                else:
                    data.add_flandmarks(None)    
            datas.add_element(data)
    return datas

def get_dataset_from_csv(csv_path):
    datas = FlandmarksDataStorage()
    file = open(csv_path)
    # the first line are title, just ignore
    file.readline()
    # Job starts here
    line = file.readline()
    EXPRESSIONS = [Exp.NEUTRAL, Exp.HAPPINESS, Exp.SURPRISE,
                   Exp.SADNESS, Exp.ANGER, Exp.DISGUST,
                   Exp.FEAR, Exp.CONTEMPT]
    while line != "":
        exp = line.split(',')[0].replace('\n','')
        exp = EXPRESSIONS[int(exp)]
        # CONTINUER ICI, LA J'AI TROP LA FLEMME
        cat = line.split(',')[2].replace('\n','')
        points = line.split(',')[1].split(' ')
        points = np.asarray(points)
        points = points.astype('uint8')
        data = FlandmarksData("no file", exp)
        # Save the n frames points
        fl = FLandmarks()
        fl.set_points(points)
        data.add_flandmarks(fl)
        datas.add_element(data)
        line = file.readline()
    return datas

def get_dataset_from_json(filepath):
    """Create a json file with the dataset.

    Param: filepath, string. Path of the file to write.
    """
    file = open(filepath, 'r')
    return json.loads(file.read())

# JSON Structs ###############################################################
# ############################################################################


def features_json(name, infos=""):
    """Return a json object for the features saving.

    Param: name, string. Name of the dataset.
    Param: infos, string. Some information on the set.
    """
    return {
            "name": name,
            "type": "FEATURES",
            "infos": infos,
            "datas": []
        }


def landmarks_json(name, infos="", normalized="NOT"):
    """Return a json object for the  landmarks saving.

    Param: name, string. Name of the dataset.
    Param: infos, string. Some information on the set.
    Param: normalized, string. Tell If the points were normalized & which way.
    """
    return {
            "name": name,
            "type": "LANDMARKS",
            "infos": infos,
            "datas": [],
            "normalized": normalized
        }


def get_landmark_points_json(shapes, cat, normalized="False"):
    """Return a list of json struct with the points informations.

    The returned list contains some objects containing an array for the landmark
    points coordinates and the boundaries of the points.

    Param: shapes, list. List of extracted shapes (see dlib).
    Param: cat, landmarks.Cat. The landmarks points category.
    Param: normalized, boolean. True if the data must be normalized.
    False otherwize.
    """
    values = []
    # Retrieve the points wanted.
    for ids, shape in enumerate(shapes):
        if shape is not None:
            values.append({
                    "points": fee.landmarks.get_points(shape, cat),
                    "bounds": fee.landmarks.get_bounds(shape, cat)
                })
        else:
            values.append(None)
    return values

# Video extractions ##########################################################
# ############################################################################

def get_n_equidistant_frames(filepath, frames_count, logs=True):
    """Return an array of n frame from a video."""
    if logs:
        print("Processing file: {}".format(filepath))
    # Open the video
    cap = cv2.VideoCapture(filepath)
    # get total number of frames
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # check for valid frame number
    if frames_count < 0 or frames_count > totalFrames:
        return None
    # Get the frames indexes
    indexes = np.linspace(0,
                          totalFrames,
                          num=frames_count,
                          dtype="int32").tolist()
    frames = []
    index = indexes.pop(0)
    # Loop over the frames
    i = 0
    while i < totalFrames:
        ret, frame = cap.read()
        if i == index:
            frames.append(frame)
            if len(indexes) == 0:
                break
            else:
                index = indexes.pop(0)
        i += 1
    cap.release()
    return frames


def get_landmarks_shapes_from_video(filepath, predictor, logs=True):
    """Return an array of landmarks shapes (see dlib) from a video."""
    if logs:
        print("Processing file: {}".format(filepath))
    cap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
    # Array for the features.
    shapes = []
    # Loop for each frame of the video.
    while(cap.isOpened()):
        if logs:
            print('.', sep='', end='', flush=True)
        ret, frame = cap.read()
        if ret:
            # Get the shapes from the frame & append the first one.
            # This would need to be refactored for multiple faces detection.
            shape = fee.landmarks.get_landmark_points(frame, predictor)
            if len(shape) > 0:
                shapes.append(shape[0])
            else:
                shapes.append(None)
        else:
            cap.release()
    if logs:
        print('\n')
    return shapes
