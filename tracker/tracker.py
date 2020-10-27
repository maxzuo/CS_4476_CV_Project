from abc import ABC, abstractmethod

class Tracker(ABC):

    def __init__(self, tracker_type):
        self.tracker = tracker_type()
        self.type = tracker_type
        pass

    def predict_frame(self, frame):
        success, bbox = self.tracker.update(frame)
        return success, bbox

    def clear(self):
        self.tracker = self.type()

    def _init_tracker(self, frame, bbox):
        self.tracker.init(frame, bbox)

    def predict_frames(self, frames, first_frame=None, bbox=None):
        frames = iter(frames)
        if bbox:
            self.clear()
            self._init_tracker(first_frame if first_frame is not None else next(frames), bbox)
        for frame in frames: yield frame, self.predict_frame(frame)