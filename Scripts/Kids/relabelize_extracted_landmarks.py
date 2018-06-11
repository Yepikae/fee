"""Copy the extracted landmarks with the reviewers labels."""

import argparse
import numpy as np
from fee.io import open_landmarks_files

# Get the paths
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source",
                    help="Landmarks source folder path.")
parser.add_argument("-t", "--target",
                    help="Landmarks target folder path.")
parser.add_argument("-l", "--labels",
                    help="Label csv file path.",
                    default=None)
args = parser.parse_args()

source_path = args.source
target_path = args.target
labels_path = args.labels

# Open the "true labels" file
f_labels = open(labels_path, 'r')
# Pick the labels from the first line, removes (%) and lowercase.
line = f_labels.readline()
labels = [e.replace('(%)', '').lower() for e in line.split(',')[24:30]]

v_labels = {}
# Put in v_labels a list fo tuple (file_name, label) for each subject (S1, ..)
line = f_labels.readline()
while line != '':
    arg_label  = np.argmax([float(e) for e in line.split(',')[24:30]])
    video_name = line.split(',')[0]
    f_landmarks = video_name.split('_')[0]+'.csv'
    if f_landmarks not in v_labels:
        v_labels[f_landmarks] = []
    v_labels[f_landmarks].append((video_name, labels[arg_label]))
    line = f_labels.readline()

# For each subject landmark csv, replace the previous label with the new one.
for f in open_landmarks_files(source_path):
    # The name of the landmark csv file is at the end of its path
    f_name = f.name.split('/')[-1]
    # If the landmarks csv file is not in v_labels, it means it's not in the
    # reviewers labels. SO we do nothing.
    if f_name not in v_labels:
        continue
    # Open the target landmarks csv file and write the headers line.
    f_target = open(target_path+f_name, 'w')
    header_line = f.readline()
    f_target.write(header_line)
    # For each line, we simply switch the expression label
    line = f.readline()
    while line != '':
        filepath = line.split(',')[0]
        remains = (',').join(line.split(',')[2:])
        for file_real_label in v_labels[f_name]:
            name, label = file_real_label
            if name in filepath:
                f_target.write(filepath+','+label+','+remains)
        line = f.readline()
