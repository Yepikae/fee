"""List of argparse options. Just because of laziness at some point."""
import argparse


def get_args(category):
    """
    Return a tuple of arguments parsed.

    Requires the category of args from the following list:
        - LANDMARKS_FROM_VIDEO
        - CREATE_NSQUARE_PICS
        - FACE_CLASSIFICATION

    Examples :

        from fee.args import get_args
        path, isRep, out, predictor = get_args('LANDMARKS_FROM_VIDEO')
    """
    if category == 'LANDMARKS_FROM_VIDEO':
        return _landmarks_from_video_()
    elif category == 'CREATE_NSQUARE_PICS':
        return _create_nsquare_pics_()
    elif category == 'FACE_CLASSIFICATION':
        return _face_classification_()
    else:
        print("The category of args <"+category+"> does not exists.")
        exit()


def _face_classification_():
    """Return the tuple(picture_path, output_path)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="Path of the pictures to classify.")
    parser.add_argument("-o", "--output",
                        help="Path of the output file.")
    parser.add_argument("-m", "--model",
                        help="Path of the model hdf5 file.")
    args = parser.parse_args()
    return (args.path, args.output, args.model)


def _landmarks_from_video_():
    """Return the tuple (path, repository, output, predictor)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="Path of the video(s) to extract.")
    parser.add_argument("-o", "--output",
                        help="Path of the output file.")
    parser.add_argument("-r", "--repository",
                        help="(optionnal) Extract the landmarks from the" +
                        "videos of a repository.",
                        action="store_true",
                        default=False)
    parser.add_argument("--predictor",
                        help="(optionnal) Path of the dlib predictor",
                        default=None)
    args = parser.parse_args()
    return (args.path, args.output, args.repository, args.predictor)


def _create_nsquare_pics_():
    """Return the tuple (path, (size, size), offset)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path",
                        help="Path of landmarks csv files.")
    parser.add_argument("-t", "--target",
                        help="Target path of the output pictures folder.")
    parser.add_argument("-s", "--size",
                        help="(optionnal) Size of the picture. Default: 64",
                        type=int, default=64)
    parser.add_argument("-os", "--offset",
                        help="(optionnal) Size of the offset between landmar" +
                             "ks bound et actual picture cropped. Default: 15",
                        type=int, default=15)
    args = parser.parse_args()
    return (args.path, args.target, (args.size, args.size), args.offset)
