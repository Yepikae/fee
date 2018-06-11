"""Extract the landmarks points from the kids database videos.

Put them in a csv file.
"""
import dlib
import os
import re
import glob
from fee.args import get_args
from fee.classification import Expression as Exp
from fee.io import get_video_landmarks
from fee.io import print_csv_line

# Get the input, output & predictor paths.
path, out, isRep, predictor = get_args('LANDMARKS_FROM_VIDEO')


# File type of video to extract
file_type = 'mp4'

# Create a global predictor (noneed to create one for each video)
if predictor is None:
    predictor = dlib.shape_predictor('./fee/sp68fl.dat')

# Sets first line of output file if necessary
output_file = open(out, 'a')
if os.stat(out).st_size == 0:
    output_file.write('file,expressions,frame,bounds,points')
    output_file.write('\n')


def record_landmarks(path):
    """Parse and record a video file landmarks."""
    # Extract expression label from video path
    val = re.findall(r"\_([a-z]*)", path, re.I)
    expressions = []
    for gid in range(0, len(val)):
        exp = Exp.from_str(val[gid])
        if exp is not None:
            expressions.append(exp)
    for id, lm in get_video_landmarks(path, predictor):
        print_csv_line(output_file,
                       [('source_file'      , path               ),
                        ('expressions'      , expressions        ),
                        ('frame_id'         , id                 ),
                        ('flandmarks_bounds', lm.get_bounds()    ),
                        ('flandmarks_points', lm.get_all_points())])


# Record the video(s) landmarks here.
if not isRep:
    record_landmarks(path)
else:
    for id, f in enumerate(glob.glob(os.path.join(path, "*."+file_type))):
        record_landmarks(f)
