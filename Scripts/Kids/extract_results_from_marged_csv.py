import argparse
from fee.classification import Expression as Exp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source",
                    help="Path of the CNN results.")
parser.add_argument("-o", "--output",
                    help="Path of the output file.")
args = parser.parse_args()

source_path = args.source
output_path = args.output

source_file = open(source_path, 'r')
output_file = open(output_path, 'w')

# The cnn labels will also be the output file labels
source_labels = [Exp.ANGER, Exp.DISGUST, Exp.FEAR, Exp.HAPPINESS, Exp.SADNESS,
                 Exp.SURPRISE, Exp.NEUTRAL]

cols = len(source_labels)
cnn_matrix = np.zeros((cols, cols))
reviewers_matrix = np.zeros((cols, cols))
total_per_exp = np.zeros(cols)


# Ignore the first line (headers) of the cnn_file
line = source_file.readline()


def get_datas_from_line(line):
    """Doc to do."""
    line = line.split(',')
    values = [float(v)*100 for v in line[1:]]
    # values = ["%.2f" % round(v, 2) for v in values]
    return values


line = source_file.readline()
while line != '':
    cnn_values = get_datas_from_line(line)
    line = source_file.readline()
    reviewers_values = get_datas_from_line(line)
    exp_arg = np.argmax(reviewers_values)
    cols = len(source_labels)
    for i in range(0, cols):
        cnn_matrix[exp_arg][i]       += cnn_values[i]
        reviewers_matrix[exp_arg][i] += reviewers_values[i]
    total_per_exp[exp_arg] += 100
    line = source_file.readline()

for i in range(0, cols):
    for j in range(0, cols):
        cnn_matrix[i][j]       /= total_per_exp[i]
        reviewers_matrix[i][j] /= total_per_exp[i]
    cnn_matrix[i] = ["%.2f" % round(v, 2) for v in cnn_matrix[i]]
    reviewers_matrix[i] = ["%.2f" % round(v, 2) for v in reviewers_matrix[i]]

source_labels = [e.to_str() for e in source_labels]
cnn_matrix = cnn_matrix.astype('str')
reviewers_matrix = reviewers_matrix.astype('str')
output_file.write('# Cnn Results \n')
output_file.write('source\target,')
output_file.write((',').join(source_labels)+'\n')
for i in range(0, cols):
    output_file.write(source_labels[i]+','+(',').join(cnn_matrix[i])+'\n')
output_file.write('# Reviewers Results \n')
output_file.write('source\target,')
output_file.write((',').join(source_labels)+'\n')
for i in range(0, cols):
    output_file.write(source_labels[i]+','+(',').join(reviewers_matrix[i])+'\n')
