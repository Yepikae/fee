## Facial Expression Extraction ##

Intro


### Dependencies ###
---
 * Python 3x
 * DLIB
 * OpenCV
 * Numpy


### Project Architecture ###
---

Here's what is in this repository.

#### Library ####

The folder `Lib` includes a few python utilities:
  * `args.py` A set of preset argument parser, just to avoid some copy past.
  * `classification.py` A set of Expression enums.
  * `data.py` Data management module. Import, manage and format the datas
  (mostly images) used for the models training/predictions.
  * `io.py` Aim to read/write files related to the `Lib` modules, such as csv
  files containing the landmarks points extracted from videos.
  * `landmarks.py` Extract the landmarks points from images or videos.
  * `models.py` A set of model constructors used in this project.
  * `recorder.py` Legacy, work in progress...
  * `sp68fl.dat` Shape_predictor_68_face_landmarks
  from https://github.com/davisking/dlib-models.
  * `utils.py` A set of genereic functions.
  * `visualizer.py` A set of function to "print" the landmarks points.

  #### Scripts ####

  The folder `Scripts/Kids` contains all the scripts and results regarding the
  work on classifying facial expression from kids pictures.
