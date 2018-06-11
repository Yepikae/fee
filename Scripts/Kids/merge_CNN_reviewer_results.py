import argparse
from fee.classification import Expression as Exp

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cnns",
                    help="Path of the CNN results.")
parser.add_argument("-r", "--reviewers",
                    help="Path of the reviewers results.")
parser.add_argument("-o", "--output",
                    help="Path of the output file.")
args = parser.parse_args()

cnn_path       = args.cnns
reviewers_path = args.reviewers
output_path    = args.output

cnn_file       = open(cnn_path, 'r')
reviewers_file = open(reviewers_path, 'r')
output_file    = open(output_path, 'w')

# The cnn labels will also be the output file labels
cnn_labels = [Exp.ANGER, Exp.DISGUST, Exp.FEAR, Exp.HAPPINESS, Exp.SADNESS,
              Exp.SURPRISE, Exp.NEUTRAL]

file_values_map = {}

# Ignore the first line (headers) of the cnn_file
line = cnn_file.readline()

line = cnn_file.readline()
while line != '':
    line = line.split(',')
    file = line[0]
    file = file.split('/')[-1]
    values = [float(v)*100 for v in line[1:]]
    values = ["%.2f" % round(v, 2) for v in values]
    file_values_map[file] = [values]
    line = cnn_file.readline()


# Parse the reviewers labels
line = reviewers_file.readline()
reviewers_labels = line.split(',')[24:30]
reviewers_labels = [s.replace('(%)', '') for s in reviewers_labels]
reviewers_labels = [Exp.from_str(s) for s in reviewers_labels]
# We now add the NEUTRAL label to replace the "Undecided" label, for more
# consistance with the cnn labels.
reviewers_labels.append(Exp.NEUTRAL)

line = reviewers_file.readline()
while line != '':
    line = line.split(',')
    file = line[0]
    values = [float(v) for v in line[24:31]]
    values = ["%.2f" % round(v, 2) for v in values]
    if file in file_values_map:
        file_values_map[file].append(values)
    line = reviewers_file.readline()

output_file.write('file')
for l in cnn_labels:
    output_file.write(','+l.to_str())
output_file.write('\n')

for f in file_values_map:
    output_file.write(f)
    for v in file_values_map[f][0]:
        output_file.write(','+v)
    output_file.write('\n')
    output_file.write(f)
    for i, l in enumerate(cnn_labels):
        index = reviewers_labels.index(l)
        output_file.write(','+file_values_map[f][1][index])
    output_file.write('\n')
output_file.close()
