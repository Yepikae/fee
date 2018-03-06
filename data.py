"""Data managment module."""
import random
import numpy as np
from fee.classification import Expression as Exp
from keras.utils import to_categorical


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

    def generate_training_set(data, expression):
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
