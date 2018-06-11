"""Add expression found at any time in a video file."""
import cv2
import dlib
from fee.classification import Expression as Exp
from fee.landmarks import get_landmark_points
from fee.landmarks import get_all_points
from fee.utils import get_bounds
from keras.models import load_model
import numpy as np
import sys

# Little hack that I don't understand
from keras import backend as K
K.set_learning_phase(1)  # Set learning phase

# = Functions =================================================================
# =============================================================================


def get_face(frame, predictor, offset):
    """Return a 68 per 68 picture fo the face of a subject."""
    shapes = get_landmark_points(frame, predictor)
    if len(shapes) == 0:
        return None
    points = get_all_points(shapes[0])
    bounds = get_bounds(points)
    top_margin = bounds["width"] - bounds["height"]
    frame = frame[bounds["top"] - offset - top_margin:
                  bounds["top"] + bounds["height"] + offset,
                  bounds["left"] - offset:
                  bounds["left"] + bounds["width"] + offset]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (64, 64))
    frame = frame.astype('float32')
    frame = frame / 255.0
    frame = frame - 0.5
    frame = frame * 2.0
    # frame = np.expand_dims(frame, 0)
    frame = np.expand_dims(frame, -1)
    return frame


# = Run =======================================================================
# =============================================================================

# We ask for the input video pth and the output video path.
if len(sys.argv) < 3:
    print('Gib video input & output paths.')
    exit()

input_path = sys.argv[1]
output_path = sys.argv[2]

# Create Dlib predictor to retrieve the landmark points
# The landmark points will allow a more precise face detection
predictorpath = './fee/sp68fl.dat'
predictor = dlib.shape_predictor(predictorpath)

# Open the input and output videos
INPUT = cv2.VideoCapture(input_path, cv2.CAP_FFMPEG)
OUTPUT = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"XVID"),
                         30, (800, 600))

# Expression list (in the right order...)
EXPS = [Exp.ANGER, Exp.DISGUST, Exp.FEAR, Exp.HAPPINESS, Exp.SADNESS,
        Exp.SURPRISE, Exp.NEUTRAL]

# Number of frame per lstm classification
FRAME_COUNT = 20

# Number of frame slide per classification
FRAME_SLIDE = 5

# Frame list
FRAMES = []
FACES = []
EMOTIONS = []
for i in range(0, FRAME_COUNT):
    EMOTIONS.append([])

# Import the model
print("LOAD MODEL")
model = load_model('./cnn_lstm_ck.hdf5')

count = 0
print("START WHILE LOOP")
while(INPUT.isOpened()):
    ret, frame = INPUT.read()
    if ret:
        if len(FRAMES) < FRAME_COUNT:
            face = get_face(frame, predictor, 15)
            if face is not None:
                FRAMES.append(frame)
                FACES.append(face)
        else:
            print("Shunk done")
            faces = np.asarray(FACES)
            # Predict the expression
            predictions = model.predict(np.asarray([faces]))
            # Get the max expression
            arg = np.argmax(predictions, axis=1)[0]
            # Write it down on the frames and save them
            for i in range(0, FRAME_SLIDE):
                EMOTIONS[i].append(arg)
                counts = np.bincount(EMOTIONS[i])
                exp = EXPS[np.argmax(counts)].to_str()
                cv2.putText(FRAMES[i], exp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 255, 255), 2)
                OUTPUT.write(FRAMES[i])
            EMOTIONS = EMOTIONS[5:]
            for i in range(0, FRAME_SLIDE):
                EMOTIONS.append([])
            for i in range(FRAME_SLIDE, len(FRAMES)):
                EMOTIONS[i].append(arg)
            # Destroy the n first frames
            FRAMES = FRAMES[FRAME_SLIDE:]
            FRAMES.append(frame)
            FACES = FACES[FRAME_SLIDE:]
            FACES.append(get_face(frame, predictor, 15))
        count += 1
    else:
        INPUT.release()
        OUTPUT.release()
