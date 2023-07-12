from utils.visualization import BBoxWithImagesVisualization
import cv2

class OpenCVImagesVisualizer:
    def __init__(self, classes_reader):
        self._visualizer = BBoxWithImagesVisualization(classes_reader)

    def draw_objects(self, orig_image, objects):
        boxes, scores, classes = objects
        self._visualizer.draw_bboxes(orig_image, boxes, scores, classes)


class OpencvJpegStorer:
    def __init__(self, starting_dir):
        self._starting_dir = starting_dir

    def store(self, frame):
        cv2.imwrite(self._starting_dir / f"img{self._frame_cntr:05}.jpg", frame)