"""Data managment module."""
import random
import numpy as np
from fee.classification import Expression as Exp
from keras.utils import to_categorical
import glob
import os
import cv2


class FlandmarksData:
    """Facial landmarks data."""

    def __init__(self, filepath, expression):
        """Constructor."""
        self.__flandmarks__ = []
        self.__filepath__ = filepath
        self.__expression__ = expression

    def get_expression(self):
        """Return __expression__."""
        return self.__expression__

    def get_flandmarks(self, index):
        """Return a flandmarks at given index."""
        return self.__flandmarks__[index]

    def get_flandmarks_length(self):
        """Return the length of __flandmarks__."""
        return len(self.__flandmarks__)

    def add_flandmarks(self, flandmarks):
        """Add an element to __flandmarks__."""
        self.__flandmarks__.append(flandmarks)

    def set_flandmarks(self, flandmarks):
        """Set __flandmarks__."""
        self.__flandmarks__ = list(flandmarks)


class FlandmarksDataStorage:
    """
    Facial landmarks storage main class.

    Store a set of Flandmarks along with their expression class and their file
    path.
    """

    def __init__(self):
        """Constructor."""
        self.__flandmarks__ = []
        self.__validations__ = []
        self.__training__ = []
        self.__expressions__ = []
        self.__validations_indexes__ = []

    def add_element(self, data):
        """Add an element to __flandmarks__."""
        self.__flandmarks__.append(data)

    def shuffle(self):
        """Shuffle __flandmarks__."""
        random.shuffle(self.__flandmarks__)

    def generate_validation_set(self, count, expressions):
        """
        Fill __validations__.

        Count is an excluded bound.
        """
        self.__expressions__ = expressions
        # Expression counters, to fill the validation dataset
        counters = {}
        for i, e in enumerate(expressions):
            counters[e] = 0
        # Loop over the data to retrieve 'count' of each category
        for i, data in enumerate(self.__flandmarks__):
            expression = data.get_expression()
            if expression in counters and counters[expression] < count:
                data = self.__flandmarks__[i]
                self.__validations__.append(data)
                counters[expression] += 1
                self.__validations_indexes__.append(i)

    def generate_training_set(self):
        """Fill __training__."""
        # Loop over the data to retrieve 'count' of each category
        for i, data in enumerate(self.__flandmarks__):
            if i not in self.__validations_indexes__:
                expression = data.get_expression()
                if expression in self.__expressions__:
                    data = self.__flandmarks__[i]
                    self.__training__.append(data)

    def reset(self):
        """Reset the sets."""
        self.__validations__ = []
        self.__training__ = []
        self.__validations_indexes__ = []

    def get_training_classes(self):
        """Return an ordered list of facial Expression."""
        result = []
        for i, data in enumerate(self.__training__):
            result.append(data.get_expression())
        return result

    # @profile(precision=8)
    def get_formated_frame_set(self,
                               frame_width,
                               frame_height,
                               frame_centered,
                               one_pixel_precision=False,
                               use_validation_set=False,
                               equally_spaced_count=None,
                               start_index=0,
                               end_index=None,
                               frames_count=None,
                               reversed=False):
        """
        Return a tuple of set.

        First one is X set (the frames)
        Second one is Y set (the classes)
        """
        XSET = []
        YSET = []
        src = []
        if use_validation_set:
            src = self.__validations__
        else:
            src = self.__training__
        for i, d in enumerate(src):
            # Retrieve the indexes of the frame we want to extract
            indexes = self.__get_indexes__(d, equally_spaced_count,
                                           start_index, end_index,
                                           frames_count, reversed)
            # For each landmarks points, extract the frames and format 'em.
            for j, index in enumerate(indexes):
                fl = d.get_flandmarks(index)
                frame = fl.get_frame(frame_width,
                                     frame_height,
                                     frame_centered,
                                     one_pixel_precision)
                frame = frame.reshape(frame_width*frame_height)
                frame = frame.astype('int8') / 255
                frame = frame.tolist()
                # framelist = self._YOLOTOLIST_(frame
                XSET.append(frame)
                YSET.append(self.__expressions__.index(d.get_expression()))
        # If we're not in binary case, we format the class datas.
        # Otherwise 0, 1 datas will do the job.
        if len(self.__expressions__) > 2:
            YSET = to_categorical(YSET)
        # del XSET
        # del YSET
        # return 0, 0
        return XSET, YSET

    def get_lstm_formated_frame_set(self,
                                    frame_width,
                                    frame_height,
                                    frame_centered,
                                    use_validation_set=False,
                                    equally_spaced_count=None,
                                    start_index=0,
                                    end_index=None,
                                    frames_count=None,
                                    reversed=False):
        """
        Return a tuple of set.

        First one is X set (the frames)
        Second one is Y set (the classes)
        """
        XSET = []
        YSET = []
        src = []
        if use_validation_set:
            src = self.__validations__
        else:
            src = self.__training__
        for i, d in enumerate(src):
            # Retrieve the indexes of the frame we want to extract
            indexes = self.__get_indexes__(d, equally_spaced_count,
                                           start_index, end_index,
                                           frames_count, reversed)
            # For each landmarks points, extract the frames and format 'em.
            sequence = []
            for j, index in enumerate(indexes):
                frame = d.get_flandmarks(index).get_frame(frame_width,
                                                          frame_height,
                                                          frame_centered)
                frame = frame.reshape(frame_width*frame_height)
                frame = frame.astype('float32') / 255
                frame = frame.tolist()
                sequence.append(frame)
            XSET.append(sequence)
            YSET.append(self.__expressions__.index(d.get_expression()))
        # If we're not in binary case, we format the class datas.
        # Otherwise 0, 1 datas will do the job.
        if len(self.__expressions__) > 2:
            YSET = to_categorical(YSET)
        return XSET, YSET

    def __get_indexes__(self, data, esc, si, ei, fc, r):
        """Return an indexes list for get_formated_validation_set."""
        indexes = []
        if ei is None:
            ei = data.get_flandmarks_length()
        if fc is None:
            fc = ei
        # Get the indexes of equally spaced frame if necessary
        if esc is not None:
            indexes = np.linspace(si, ei-1, num=esc, dtype="int32")
        # Otherwise fill indexes with the indexes (yeah, could be prettier)
        else:
            if reversed:
                for i in range(ei-1, ei-fc-1, -1):
                    indexes.append(i)
            else:
                for i in range(si, fc):
                    indexes.append(i)
        return indexes


class FER2013Dataset:
    """FER2013 data storage and management."""

    def __init__(self):
        """Constructor."""
        self.__training__ = {
                Exp.ANGER: [],
                Exp.DISGUST: [],
                Exp.FEAR: [],
                Exp.HAPPINESS: [],
                Exp.SADNESS: [],
                Exp.SURPRISE: [],
                Exp.NEUTRAL: []
            }
        self.__validation__ = {
                Exp.ANGER: [],
                Exp.DISGUST: [],
                Exp.FEAR: [],
                Exp.HAPPINESS: [],
                Exp.SADNESS: [],
                Exp.SURPRISE: [],
                Exp.NEUTRAL: []
            }
        self.__expressions__ = [Exp.ANGER, Exp.DISGUST, Exp.FEAR,
                                Exp.HAPPINESS, Exp.SADNESS, Exp.SURPRISE,
                                Exp.NEUTRAL]

    def load_csv(self, filepath, ignore_first=True):
        """Fill __pictures__ with csv datas."""
        file = open(filepath)
        if ignore_first:
            file.readline()
        line = file.readline()
        while line != '':
            # Split the line
            cells = line.split(',')
            # Retrieve the pixels
            pixels = cells[1].split(' ')
            pixels = np.asarray(pixels)
            pixels = pixels.astype('uint8')
            pixels = pixels.reshape(48, 48, 1)
            # Retrieve the expression
            exp_arg = int(cells[0])
            exp = self.__expressions__[exp_arg]
            # Retrieve th target set
            set_arg = cells[2].replace('\n', '')
            if set_arg.lower() == "training":
                self.__training__[exp].append(pixels)
            elif set_arg.lower() == "publictest":
                self.__validation__[exp].append(pixels)
            elif set_arg.lower() == "privatetest":
                self.__validation__[exp].append(pixels)
            line = file.readline()

    def shuffle(self):
        """Shuffle __pictures__."""
        for exp in self.__training__:
            random.shuffle(self.__training__[exp])
        for exp in self.__validation__:
            random.shuffle(self.__validation__[exp])

    def get_n_pictures(self, expressions, target="training",
                       count=0, start=0):
        """Return a list of n pictures per expressions."""
        piclist = []
        tmplist = self.__training__
        if target == "validation":
            tmplist = self.__validation__
        # For each expression
        for i, exp in enumerate(expressions):
            # If there still is some expression
            if start < len(tmplist[exp]):
                # We pick at least the expressions remaining
                end = start+count
                if end > len(tmplist[exp]):
                    end = len(tmplist[exp])
                for j in range(start, end):
                    piclist.append((tmplist[exp][j], exp))
        return piclist

    def get_pictures(self, expressions, target="training",
                     part_size=1, part_index=0):
        """Return a list of pictures."""
        piclist = []
        tmplist = self.__training__
        if target == "validation":
            tmplist = self.__validation__
        # For each expression
        for i, exp in enumerate(expressions):
            # If there still is some expression
            start = int(part_index*part_size*len(tmplist[exp]))
            # We pick at least the expressions remaining
            end = start + int(len(tmplist[exp])*part_size)
            if end > len(tmplist[exp]):
                end = len(tmplist[exp])
            for j in range(start, end):
                piclist.append((tmplist[exp][j], exp))
        return piclist

    def get_data_length(self, expression, target="training"):
        """Return the number of pictures for an expression."""
        if target == "training":
            return len(self.__training__[expression])
        return len(self.__validation__[expression])


class ComparisonClassifGen:
    """Generator of comparison classification neural network sets."""

    def __init__(self):
        """Constructor."""

    def generate_training_set(data):
        """Generate a training set."""
        X1_DATA = []
        X2_DATA = []
        Y_DATA = []
        for i in range(0, len(data)):
            pic1, exp1 = data[i]
            for j in range(0, len(data)):
                if j != i:
                    pic2, exp2 = data[j]
                    X1_DATA.append(pic1)
                    X2_DATA.append(pic2)
                    if exp1 is not exp2:
                        Y_DATA.append(0)
                    else:
                        Y_DATA.append(1)
        print("LEN X1_DATA : "+str(len(X1_DATA)))
        X1 = np.array(X1_DATA)
        X2 = np.array(X2_DATA)
        return X1, X2, Y_DATA

    def generate_monoclass_training_set(data, expression):
        """Generate a training set."""
        X1_DATA = []
        X2_DATA = []
        Y_DATA = []
        for i in range(0, len(data)):
            pic1, exp1 = data[i]
            if exp1 is expression:
                for j in range(i+1, len(data)):
                    pic2, exp2 = data[j]
                    X1_DATA.append(pic1)
                    X2_DATA.append(pic2)
                    if exp1 is not exp2:
                        Y_DATA.append(0)
                    else:
                        Y_DATA.append(1)
        print("LEN X1_DATA : "+str(len(X1_DATA)))
        X1 = np.array(X1_DATA)
        X2 = np.array(X2_DATA)
        return X1, X2, Y_DATA


class SimpleClassGen:
    """Generate simple set of data for DL models."""

    def __init__(self):
        """Contructor."""

    def generate_sets_facenet(self, data, num_classes=None):
        """
        Generate the facenet models compatible sets.

        data is a list of tuples (picture, class ID).
        """
        X = []
        Y = []
        for i in range(0, len(data)):
            pic, exp = data[i]
            pic = pic.astype('float32')
            pic = pic / 255.0
            pic = pic - 0.5
            pic = pic * 2.0
            # pic = np.expand_dims(pic, 0)
            pic = np.expand_dims(pic, -1)
            X.append(pic)
            Y.append(exp)
        X = np.asarray(X)
        Y = to_categorical(Y, num_classes=num_classes)
        return X, Y

    def __get_indexes__(self, data, esc, si, ei, fc, r):
        """Return an indexes list for get_formated_validation_set."""
        indexes = []
        if ei is None:
            ei = data.get_flandmarks_length()
        if fc is None:
            fc = ei
        # Get the indexes of equally spaced frame if necessary
        if esc is not None:
            indexes = np.linspace(si, ei-1, num=esc, dtype="int32")
        # Otherwise fill indexes with the indexes (yeah, could be prettier)
        else:
            if reversed:
                for i in range(ei-1, ei-fc-1, -1):
                    indexes.append(i)
            else:
                for i in range(si, fc):
                    indexes.append(i)
        return indexes

    def generate_video_sets_facenet(self, data, frame_limit):
        """
        Generate the facenet models compatible sets.

        data is a list of tuples (picture, class ID).
        """
        X = []
        Y = []
        for i in range(0, len(data)):
            pics, exp = data[i]
            if len(pics[5:]) >= frame_limit:
                pics = pics[5:]
                # we get as many "frame_limit" long video as we can.
                pos = len(pics) - 1
                while(pos-20 >= 0):
                    tmpX = []
                    for j in range(pos, pos-20, -1):
                        pic = pics[j]
                        pic = pic.astype('float32')
                        pic = pic / 255.0
                        pic = pic - 0.5
                        pic = pic * 2.0
                        # pic = np.expand_dims(pic, 0)
                        pic = np.expand_dims(pic, -1)
                        tmpX.append(pic)
                    tmpX = np.asarray(tmpX)
                    X.append(tmpX)
                    Y.append(exp)
                    pos -= 5
        unique, counts = np.unique(Y, return_counts=True)
        print(dict(zip(unique, counts)))
        X = np.asarray(X)
        print(X.shape)
        Y = to_categorical(Y)
        return X, Y


class PicturesDataset:
    """Datatset loading and sorting pictures from different folders."""

    def __init__(self, folderpath):
        """Constructor."""
        self.__trainingpath__ = None
        self.__evaluatepath__ = None
        self.__expressions__ = []
        self.__trainingset__ = {}
        self.__evaluateset__ = {}
        subfolders = os.listdir(folderpath)
        # Check for trailing '/'
        if folderpath[-1:] != '/':
            folderpath += '/'
        self.__trainingpath__ = folderpath
        # If we got the training & evaluate subfolders
        if "training" in subfolders:
            self.__trainingpath__ = folderpath+"training/"
            if "evaluate" in subfolders:
                self.__evaluatepath__ = folderpath+"evaluate/"
        # get the classes
        exps = os.listdir(self.__trainingpath__)
        for i, e in enumerate(exps):
            if Exp.from_str(e) is None:
                print("Unknown expression: "+e)
            else:
                self.__expressions__.append(Exp.from_str(e))
        # Init the sets
        for i, e in enumerate(self.__expressions__):
            self.__trainingset__[e] = []
            self.__evaluateset__[e] = []

    def load_pictures(self):
        """Load pictures from the tree structure."""
        exps = os.listdir(self.__trainingpath__)
        for i, e in enumerate(exps):
            print(e)
            if Exp.from_str(e) is None:
                print("Unknown expression: "+e)
            else:
                exp = Exp.from_str(e)
                path = self.__trainingpath__+e+'/'
                source_map = open(path+"map.csv")
                line = source_map.readline()    # First line, headers, ignore.
                line = source_map.readline()
                while line != '':
                    line = line.split(',')
                    pic = cv2.imread(line[1].replace('\n', ''))
                    pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                    self.__trainingset__[exp].append((pic, line[0]))
                    line = source_map.readline()
        if self.__evaluatepath__ is not None:
            exps = os.listdir(self.__evaluatepath__)
            for i, e in enumerate(exps):
                if Exp.from_str(e) is None:
                    print("Unknown expression: "+e)
                else:
                    exp = Exp.from_str(e)
                    path = self.__evaluatepath__+e+'/'
                    for id, f in enumerate(glob.glob(os.path.join(path,
                                                                  "*.jpg"))):
                        source_map = open(path+"map.csv")
                        line = source_map.readline()
                        line = source_map.readline()
                        while line != '':
                            line = line.split(',')
                            pic = cv2.imread(line[1].replace('\n', ''))
                            pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
                            self.__evaluateset__[exp].append((pic, line[0]))
                            line = source_map.readline()

    def shuffle(self):
        """Shuffle the sets."""
        for i, e in enumerate(self.__trainingset__):
            random.shuffle(self.__trainingset__[e])

    def split_set(self, count):
        """Split the training set to fill the evaluate set."""
        for i, e in enumerate(self.__trainingset__):
            self.__evaluateset__[e] = self.__trainingset__[e][0:count]
            self.__trainingset__[e] = self.__trainingset__[e][count:]

    def get_pics(self, set="training", expressions=None, get_filepaths=False):
        """
        Return tuples (pictures, class ID).

        Class Id is according position in __expressions__.
        """
        result = []
        filepaths = []
        pics = self.__trainingset__
        if expressions is None:
            expressions = self.__expressions__
        if set != "training":
            pics = self.__evaluateset__
        for i, e in enumerate(pics):
            if e in expressions:
                for j in range(0, len(pics[e])):
                    pic, file = pics[e][j]
                    result.append((pic, expressions.index(e)))
                    if get_filepaths:
                        filepaths.append(file)
        if get_filepaths:
            return result, filepaths
        return result


class VideosDataset:
    """Datatset loading and sorting videos from different folders."""

    def __init__(self, folderpath):
        """Constructor."""
        self.__trainingpath__ = None
        self.__evaluatepath__ = None
        self.__expressions__ = []
        self.__trainingset__ = {}
        self.__evaluateset__ = {}
        subfolders = os.listdir(folderpath)
        # Check for trailing '/'
        if folderpath[-1:] != '/':
            folderpath += '/'
        self.__trainingpath__ = folderpath
        # If we got the training & evaluate subfolders
        if "training" in subfolders:
            self.__trainingpath__ = folderpath+"training/"
            if "evaluate" in subfolders:
                self.__evaluatepath__ = folderpath+"evaluate/"
        # get the classes
        exps = os.listdir(self.__trainingpath__)
        for i, e in enumerate(exps):
            if Exp.from_str(e) is None:
                print("Unknown expression: "+e)
            else:
                self.__expressions__.append(Exp.from_str(e))
        # Init the sets
        for i, e in enumerate(self.__expressions__):
            self.__trainingset__[e] = []
            self.__evaluateset__[e] = []

    def load_videos(self):
        """Load pictures from the tree structure."""
        exps = os.listdir(self.__trainingpath__)
        for i, e in enumerate(exps):
            if Exp.from_str(e) is None:
                print("Unknown expression: "+e)
            else:
                exp = Exp.from_str(e)
                path = self.__trainingpath__+e+'/'
                for id, f in enumerate(glob.glob(os.path.join(path, "*.avi"))):
                    pics = []
                    cap = cv2.VideoCapture(f, cv2.CAP_FFMPEG)
                    while(cap.isOpened()):
                        ret, frame = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            pics.append(frame)
                        else:
                            cap.release()
                    self.__trainingset__[exp].append(pics)
        if self.__evaluatepath__ is not None:
            exps = os.listdir(self.__evaluatepath__)
            for i, e in enumerate(exps):
                if Exp.from_str(e) is None:
                    print("Unknown expression: "+e)
                else:
                    exp = Exp.from_str(e)
                    path = self.__evaluatepath__+e+'/'
                    for id, f in enumerate(glob.glob(os.path.join(path,
                                                                  "*.avi"))):
                        pics = []
                        cap = cv2.VideoCapture(f, cv2.CAP_FFMPEG)
                        while(cap.isOpened()):
                            ret, frame = cap.read()
                            if ret:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                pics.append(frame)
                            else:
                                cap.release()
                        self.__trainingset__[exp].append(pics)

    def shuffle(self):
        """Shuffle the sets."""
        for i, e in enumerate(self.__trainingset__):
            random.shuffle(self.__trainingset__[e])

    def split_set(self, count):
        """Split the training set to fill the evaluate set."""
        for i, e in enumerate(self.__trainingset__):
            self.__evaluateset__[e] = self.__trainingset__[e][0:count]
            self.__trainingset__[e] = self.__trainingset__[e][count:]

    def get_videos(self, set="training", expressions=None):
        """
        Return tuples (pictures, class ID).

        Class Id is according position in __expressions__.
        """
        result = []
        pics = self.__trainingset__
        if expressions is None:
            expressions = self.__expressions__
        if set != "training":
            pics = self.__evaluateset__
        for i, e in enumerate(pics):
            for j in range(0, len(pics[e])):
                result.append((pics[e][j],
                               expressions.index(e)))
        return result
