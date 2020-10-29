from abc import ABC
import timeit

class Tracker(ABC):

    def __init__(self, tracker_type, timed=False):
        self.type = tracker_type
        self.tracker = self.type()
        self.timed = timed

        if timed:
            self.time = 0.
            self.frames = 0
            self.predict_frame = self._timed_predict_frame
        else:
            self.__getattribute__ = self._alt_getattr

    def _alt_getattr(self, attr):
        if attr == "_timed_predict_frame" or attr == "avg_time_per_frame":
            raise AttributeError

        return object.__getattribute__(self, attr)

    # invisible if timed is set to False
    def avg_time_per_frame(self):
        return self.time / self.frames

    # same as _timed_predict_frame without the timing overhead
    def predict_frame(self, frame):
        success, bbox = self.tracker.update(frame)
        return success, bbox

    # invisible if timed is set to False
    def _timed_predict_frame(self, frame):
        t = timeit.default_timer()
        success, bbox = self.tracker.update(frame)

        self.time += timeit.default_timer() - t
        self.frames += 1

        return success, bbox

    def clear(self):
        del self.tracker
        self.tracker = self.type()

    def _init_tracker(self, frame, bbox):
        bbox = tuple(map(int, bbox))
        return self.tracker.init(frame, bbox)

    def predict_frames(self, frames, first_frame=None, bbox=None):
        frames = iter(frames)
        if bbox:
            self.clear()
            self._init_tracker(first_frame if first_frame is not None else next(frames), bbox)
        for frame in frames: yield frame, self.predict_frame(frame)