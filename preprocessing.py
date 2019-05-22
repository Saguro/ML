import numpy as np


class MovingAveragePreprocessor:
    def __init__(self, update_factor=0.150):
        self.update_factor = update_factor
        self.avg_img = None

    def process(self, img):
        img = img - np.min(img)
        img = img / (np.max(img) + 0.00001)
        if self.avg_img is None:
            self.avg_img = img
        self.avg_img = (1 - self.update_factor) * self.avg_img + self.update_factor * img
        self.avg_img = self.avg_img - np.min(self.avg_img)
        self.avg_img = self.avg_img / (np.max(self.avg_img) + 0.00001)
        return_frame = img - self.avg_img
        return_frame = return_frame - np.min(return_frame)
        return_frame = return_frame / (np.max(return_frame) + 0.00001)
        return return_frame


def normalize_frame(frame):
    frame = frame.astype(np.float32)
    frame = frame - np.min(frame)
    frame = frame / np.max(frame)
    return frame
