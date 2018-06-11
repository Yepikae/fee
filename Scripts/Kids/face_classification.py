"""Use the face classification model."""

from fee.args import get_args
from fee.classification import Expression as Exp
from fee.data import PicturesDataset
from fee.data import SimpleClassGen
from fee.models import BITBOTSFaceClassification as FaceClassif
import numpy as np

picture_path, output_path, model_path = get_args('FACE_CLASSIFICATION')

# Model creation/managment
model, labels = FaceClassif(model_path).get_model_and_labels()

print(labels)

# Data management
dataset = PicturesDataset(picture_path)
dataset.load_pictures()
dataset.shuffle()
pics, filepaths = dataset.get_pics(expressions=labels, get_filepaths=True)

class_gen = SimpleClassGen()
X, Y = class_gen.generate_sets_facenet(pics, num_classes=len(labels))

predictions = model.predict(X)

results = [[] for i in range(0, len(labels))]

print(' = NEUTRAL FREE ================================= ')

for i, e in enumerate(predictions):
    prediction = np.argmax(e)
    real = np.argmax(Y[i])
    if labels[prediction] == Exp.NEUTRAL or prediction != real:
        results[real].append(0)
    else:
        results[real].append(1)

for i in range(0, len(results)):
    print(np.sum(results[i]))
    accuracy = np.sum(results[i]) / len(results[i])
    print(labels[i].to_str() + ' : ' + str(accuracy))

print(' = NEUTRAL IGNORED ============================== ')

results = [[] for i in range(0, len(labels))]

for i, e in enumerate(predictions):
    prediction = np.argmax(e[:-1])
    real = np.argmax(Y[i])
    if prediction != real:
        results[real].append(0)
    else:
        results[real].append(1)

for i in range(0, len(results)):
    print(np.sum(results[i]))
    accuracy = np.sum(results[i]) / len(results[i])
    print(labels[i].to_str() + ' : ' + str(accuracy))

print(' = GROUP BY FILE ================================ ')

results = {}

for i, e in enumerate(predictions):
    filepath = filepaths[i]
    if filepath not in results:
        results[filepath] = [np.argmax(e)]
    else:
        results[filepath].append(np.argmax(e))

for f in results:
    unique, counts = np.unique(results[f], return_counts=True)
    exp_count = dict(zip(unique, [c/len(results[f]) for c in counts]))
    results[f] = exp_count

file = open(output_path, 'w')
file.write('filepath')
for l in labels:
    file.write(','+l.to_str())
file.write('\n')
for f in results:
    file.write(f)
    for i in range(0, len(labels)):
        if i in results[f]:
            file.write(','+str(results[f][i]))
        else:
            file.write(',0')
    file.write('\n')
