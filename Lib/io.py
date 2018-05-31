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
from fee.utils import print_processing


def get_picture_landmarks(filepath, predictor, logs=True):
    """
    Do the doc!
    """
    if logs:
        print("Processing file: {}".format(filepath))
    frame = cv2.imread(filepath)
    lm = FLandmarks()
    lm.extract_points(frame, predictor)
    return lm
    if logs:
        print('\n')


def get_video_landmarks(filepath, predictor, logs=True):
    """
    Generator that return a tuple of the id and the landmarks of a frame.

    Return :
     - The frame id is basically the index of the frame in the whole video.
     - The frame landmarks is a FLandmarks (see fee.landmarks) filled with the
    frame.

    Parameters :
     - filepath, the video file path. Required.
     - predictor, the dlib predictor object. Required.
     - logs, True for printing the parsing progress. False otherwise.

    Examples :

        import dlib
        from fee.io import get_video_landmarks

        # Print the bounds of the first face found in each frame of the video.
        path = './video.mp4'
        predictor = dlib.shape_predictor('./fee/sp68fl.dat')
        for id, landmarks in get_video_landmarks(path, predictor):
            print(lm.get_bounds())
    """
    if logs:
        print("Processing file: {}".format(filepath))
    # Open video file
    cap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    # Loop for each frame of the video.
    while(cap.isOpened()):
        if logs:
            print_processing(float(frame_id/frame_count))
        ret, frame = cap.read()
        if ret:
            lm = FLandmarks()
            lm.extract_points(frame, predictor)
            yield (frame_id, lm)
        else:
            cap.release()
        frame_id += 1
    if logs:
        print('\n')


def open_landmarks_files(folderpath, extension='csv', reading_mode='r'):
    """Write some doc."""
    type = '*.'+extension
    for id, f in enumerate(glob.glob(os.path.join(folderpath, type))):
        yield open(f, reading_mode)


def read_landmarks_file(file, is_video=True, ignore_first_line=True):
    """Write some doc."""
    if ignore_first_line:
        line = file.readline()
    line = file.readline()
    while line != "":
        pos = 0
        line = line.split(',')
        # The source file path
        filepath = line[pos]
        pos += 1
        # The expressions
        expressions = line[pos].split(' ')
        pos += 1
        # The frame id
        if is_video:
            frame_id = int(line[pos])
            pos += 1
        # The bounds of the landmark points.
        # If there are no bounds, it means dlib haven't found any landmarks,
        # so we return None. Otherwise, we keep parsing.
        bounds = None
        points = None
        if line[pos] != '':
            bounds = [int(i) for i in line[pos].split(' ')]
            bounds = {'left'  : bounds[0],
                      'top'   : bounds[1],
                      'width' : bounds[2],
                      'height': bounds[3]}
            pos += 1
            # The landmark points
            points = [int(i) for i in line[pos].split(' ')]
        line = file.readline()
        if is_video:
            yield (filepath, frame_id, expressions, bounds, points)
        else:
            yield (filepath, expressions, bounds, points)
    return None





def print_csv_line(file, elems):
    """
    Write a line in a csv file.

    Parameters :
     - file, the output file. Required.
     - elems, a list of tuples category & content. The following category can
     be writen on the output file :
        - 'source_file'       : the path of the source file.
        - 'expression'        : A single fee.classification.Expression
        - 'expressions'       : A list of fee.classification.Expression
        - 'frame_id'          : An integer
        - 'flandmarks_bounds' : The bounds of a fee.landmarks.FLandmarks
        - 'flandmarks_points' : The points of a fee.landmarks.FLandmarks

    Example :

        import cv2
        import dlib
        from fee.classification import Expression as Exp
        from fee.landmarks import FLandmarks
        from fee.io import print_csv_line

        # Get the landmark points from a picture
        path = './my_frame.jpeg'
        image = cv2.imread(path)
        predictor = dlib.shape_predictor('./fee/sp68fl.dat')
        lm = FLandmarks()
        lm.extract_points(image, predictor)

        # Print the landmarks in a csv file
        output_file = open('./output.csv', 'w')
        output_file.write('file,expressions,bounds,points')
        print_csv_line(output_file,
                       [('source_file'      , path                 ),
                        ('expressions'      , [Exp.FEAR, Exp.ANGER]),
                        ('flandmarks_bounds', lm.get_bounds()      ),
                        ('flandmarks_points', lm.get_all_points()  )])
    """
    s = ""
    for i, tuple in enumerate(elems):
        arg, elem = tuple
        if arg == "source_file":
            s += elem
        elif arg == "expression":
            s += elem.to_str()
        elif arg == "expressions":
            s += (' ').join(e.to_str() for e in elem)
        elif arg == "frame_id":
            s += str(elem)
        elif arg == "flandmarks_bounds":
            bounds = elem
            if bounds is not None:
                s += str(bounds["left"])+" "+str(bounds["top"])+" "
                s += str(bounds["width"])+" "+str(bounds["height"])
        elif arg == "flandmarks_points":
            s += (' ').join(str(e) for e in elem)
        else:
            print("Error: arg "+arg+" unknown in fee.io.print_csv_line.")
            exit()
        if i < len(elems) - 1:
            s += ','
    file.write(s+'\n')

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
