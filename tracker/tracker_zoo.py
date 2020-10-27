import cv2
from .tracker import Tracker

class CSRT(Tracker):
    def __init__(self,frame=None,bbox=None):
        super().__init__(cv2.TrackerCSRT_create)
        if frame is not None and bbox is not None:
            self._init_tracker(frame, bbox)

class KCF(Tracker):
    def __init__(self,frame=None,bbox=None):
        __super__().__init__(cv2.TrackerKCF_create)
        if frame is not None and bbox is not None:
            self._init_tracker(frame, bbox)

class GOTURN(Tracker):
    def __init__(self,frame=None,bbox=None):
        __super__().__init__(cv2.TrackerGOTURN_create)
        if frame is not None and bbox is not None:
            self._init_tracker(frame, bbox)