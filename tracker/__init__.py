from .tracker_zoo import *
from .video_iterator import VideoIterator

def factory(tracker_type: str, **kwargs):
    models = {
        'CSRT': CSRT,
        'KCF': KCF,
        'GOTURN': GOTURN,
    }

    model = models.get(tracker_type, None)
    if model is None:
        raise ValueError("'%s' is not a legal tracker model" % tracker_type)
    return model(**kwargs)