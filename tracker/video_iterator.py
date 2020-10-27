class VideoIterator:

    def __init__(self, video):
        if not video.isOpened():
            raise Exception("Could not open video")
        self.video = video # cv2 cap object

    def release(self):
        self.video.release()

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.video.read()

        if not ret:
            self.release()
            raise StopIteration
        else:
            return frame