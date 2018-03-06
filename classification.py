"""Todo."""

from enum import Enum


class Expression(Enum):
    """Facial expression classes."""

    HAPPINESS = 1
    FEAR = 2
    ANGER = 3
    DISGUST = 4
    SADNESS = 5
    SURPRISE = 6
    NEUTRAL = 7
    CONFUSION = 8
    CONTEMPT = 9

    def from_str(string):
        """Return the Expression enum from a string."""
        if string.lower() == "happiness":
            return Expression(1)
        elif string.lower() == "happy":
            return Expression(1)
        elif string.lower() == "fear":
            return Expression(2)
        elif string.lower() == "anger":
            return Expression(3)
        elif string.lower() == "disgust":
            return Expression(4)
        elif string.lower() == "sadness":
            return Expression(5)
        elif string.lower() == "sad":
            return Expression(5)
        elif string.lower() == "surprise":
            return Expression(6)
        elif string.lower() == "neutral":
            return Expression(7)
        elif string.lower() == "confusion":
            return Expression(8)
        elif string.lower() == "confusing":
            return Expression(8)
        elif string.lower() == "contempt":
            return Expression(9)
        else:
            return None

    def to_str(enum):
        """Return the Expression enum as a string."""
        if enum == Expression.HAPPINESS:
            return "happiness"
        elif enum == Expression.FEAR:
            return "fear"
        elif enum == Expression.ANGER:
            return "anger"
        elif enum == Expression.DISGUST:
            return "disgust"
        elif enum == Expression.SADNESS:
            return "sadness"
        elif enum == Expression.SURPRISE:
            return "surprise"
        elif enum == Expression.NEUTRAL:
            return "neutral"
        elif enum == Expression.CONFUSION:
            return "confusion"
        elif enum == Expression.CONTEMPT:
            return "contempt"
        else:
            return None

    def to_int(enum):
        """Return the Expression enum as a string."""
        if enum == Expression.HAPPINESS:
            return 1
        elif enum == Expression.FEAR:
            return 2
        elif enum == Expression.ANGER:
            return 3
        elif enum == Expression.DISGUST:
            return 4
        elif enum == Expression.SADNESS:
            return 5
        elif enum == Expression.SURPRISE:
            return 6
        elif enum == Expression.NEUTRAL:
            return 7
        elif enum == Expression.CONFUSION:
            return 8
        elif enum == Expression.CONTEMPT:
            return 9
        else:
            return None
