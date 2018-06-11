"""Create face pics from the videos."""
from fee.classification import Expression as Exp
from fee.io import open_landmarks_files
from fee.io import read_landmarks_file
from fee.args import get_args
import cv2
import numpy as np
import os


def extract_face_frame(frame, bounds, size, offset):
    """Extract and format the frame."""
    top_margin = bounds["width"] - bounds["height"]
    frame = frame[bounds["top"] - offset - top_margin:
                  bounds["top"] + bounds["height"] + offset,
                  bounds["left"] - offset:
                  bounds["left"] + bounds["width"] + offset]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (size))
    return frame


folderpath, targetpath, size, offset = get_args('CREATE_NSQUARE_PICS')

current_filepath = ""
current_capture = None

# Check if the target folder already exists and if there already are files.
label_pic_counts = {}
path_map_files = {}
d = targetpath
subfolders = [os.path.join(d, o) for o in os.listdir(d)
              if os.path.isdir(os.path.join(d, o))]
for sub in subfolders:
    name = sub.replace(targetpath, '')
    label_pic_counts[name] = len([f for f in os.listdir(sub)
                                  if os.path.isfile(os.path.join(sub, f))])
    path_map_files[name] = open(sub+'/map.csv', 'a')

for f in open_landmarks_files(folderpath):            # For each file
    for line in read_landmarks_file(f):               # For each line
        filepath, frame_id, expressions, bounds, points = line
        if bounds is None:
            continue
        # If we switched to a new video file, we open a new cv2 capture
        if filepath != current_filepath:
            current_filepath = filepath
            if current_capture is not None:
                current_capture.release()
            current_capture = cv2.VideoCapture(filepath, cv2.CAP_FFMPEG)
        # We get the current frame,a ccording to the line in the csv file
        ret, frame = current_capture.read()
        if ret:
            frame = extract_face_frame(frame, bounds, size, offset)
            if expressions[0] not in label_pic_counts:
                label_pic_counts[expressions[0]] = 0
                new_path = targetpath + expressions[0]
                os.makedirs(new_path)
                path_map_files[expressions[0]] = open(new_path+'/map.csv', 'w')
                path_map_files[expressions[0]].write('source_file,picture\n')

            s = targetpath + expressions[0] + '/'
            s += str(label_pic_counts[expressions[0]]).zfill(5)
            s += '.jpg'
            path_map_files[expressions[0]].write(filepath+','+s+'\n')
            label_pic_counts[expressions[0]] += 1
            cv2.imwrite(s, frame)
