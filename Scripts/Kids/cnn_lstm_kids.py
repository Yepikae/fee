"""Classification algorithm for the kids database.

Requirements:
    - A csv file containing the extracted landmark points from the kids
    videos.
    - A trained lstm.

Example:
    python cnn_lstm_kids.py --source ./extracted.csv --model ./lstm.hdf5
    --output ./results.csv
"""
import cv2
from fee.classification import Expression as Exp
from fee.io import open_landmarks_files
from fee.io import read_landmarks_file
from keras.models import load_model
import numpy as np
import argparse

# Little hack that I don't understand, in order to reuse an existing model.
from keras import backend as K
K.set_learning_phase(1)  # Set learning phase

# Parsing the command arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source",
                    help="Path of the kids csv files.")
parser.add_argument("-m", "--model",
                    help="Path of the trained lstm model.")
parser.add_argument("-o", "--output",
                    help="Path of the output file.")
args = parser.parse_args()

# The global paths variables
source_path = args.source   # The extracted landmarks csv file path
model_path  = args.model    # The pre-trained model path
output_path = args.output   # The output csv file path


def group_line_by_file(file):
    """Parse an extracted landmarks file.

    Read each line of a csv file and return a list of all datas from a same
    video.

    Parameters:
        file: the landmarks csv file, opened in read mode.

    Returns:
        current_file: the current returned file path
        exp: the current returned file main expression
        datas: a list of the current returned file landmarks
        (see fee.io.read_landmarks_file)
    """
    current_file = None
    datas = []
    for data in read_landmarks_file(file):
        filepath, frame_id, exp, bounds, points = data
        if current_file != filepath:
            if current_file is not None:
                yield current_file, exp, datas
            datas = []
            current_file = filepath
        datas.append(data)
    yield current_file, exp, datas


def extract_face_frame(frame, bounds, size, offset):
    """Crop and resize a frame.

    Parameters:
        frame: A frame.
        bounds: The bounding box to crop.
        size: The size to apply to the cropped part.
        offset: The offset to add to the bounding box.

    Returns:
        A frame.
    """
    top_margin = bounds["width"] - bounds["height"]
    frame = frame[bounds["top"] - offset - top_margin:
                  bounds["top"] + bounds["height"] + offset,
                  bounds["left"] - offset:
                  bounds["left"] + bounds["width"] + offset]
    frame = cv2.resize(frame, (size))
    return frame


def format_face_frame(frame):
    """Format for the face_classification model.

    Parameters:
        frame: A frame.

    Returns:
        A frame.
    """
    frame = frame.astype('float32')
    frame = frame / 255.0
    frame = frame - 0.5
    frame = frame * 2.0
    frame = np.expand_dims(frame, -1)
    return frame


# The number of frame for each sequence to classify. Must be equivalent of the
# number of frame used for the trained model.
frame_count = 20

model  = load_model(model_path)
labels = [Exp.ANGER, Exp.DISGUST, Exp.FEAR, Exp.HAPPINESS,
          Exp.SADNESS, Exp.SURPRISE, Exp.NEUTRAL]

# Write the header of the output csv file.
output_file = open(output_path, 'w')
output_file.write('filepath,'+(',').join([l.to_str() for l in labels])+'\n')

# For each extracted landmarks csv file, we do the prediction.
for f in open_landmarks_files(source_path):
    for filepath, exp, datas in group_line_by_file(f):
        exp = Exp.from_str(exp[0])
        # Retrieve the faces frame by frame
        cap = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
        frameid = 0
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                filepath, frame_id, exp, bounds, points = datas[frameid]
                # The dlib algorithm might not succeeded in finding a face.
                # In such case, the bounding box is equal to None. So we'll do
                # the face extraction on the frame which actually have a
                # bounding box.
                if bounds is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = extract_face_frame(frame, bounds, (64, 64), 15)
                    frame = format_face_frame(frame)
                    frames.append(frame)
                frameid += 1
            else:
                cap.release()
        # Predict the expression. We predict for n sequences of <frame_count>
        # frames. Each sequence starts with a slide of 5 frame from the
        # previous one.
        i = 0
        X = []
        # We fill the predict set.
        while i + frame_count < len(frames):
            X.append(np.array(frames[i:i+frame_count]))
            i += 5
        # There might be no sequence to predict, because the video is not long
        # enough (less than <frame_count> frames) or because the dlib algorithm
        # couldn't find any face.
        if len(X) > 0:
            X = np.array(X)
            print(filepath)
            prediction = model.predict(X)
            argmax = np.argmax(prediction, axis=1)
            # Uncomment the following line to remove the NEUTRAL prediction.
            # argmax = np.argmax(prediction[:, :-1], axis=1)
            unique, counts = np.unique(argmax, return_counts=True)
            counts = [c / len(argmax) for c in counts]
            results = np.zeros(len(labels))
            for i in range(0, len(unique)):
                results[unique[i]] = counts[i]
            output_file.write(filepath+',')
            output_file.write((',').join([str(r) for r in results])+'\n')
