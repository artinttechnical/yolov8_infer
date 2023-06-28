from utils.visualization import BBoxVisualization
import cv2

class OpenCVImagesVisualizer:
    def __init__(self, classes, images_path, images_suffix):
        self._visualizer = BBoxVisualization(classes, images_path, images_suffix)

    def visualize(self, orig_image, objects):
        boxes, scores, classes = objects
        self._visualizer.draw_bboxes(orig_image, boxes, scores, classes)


class OpencvJpegStorer:
    def __init__(self, starting_dir):
        self._starting_dir = starting_dir

    def store(self, frame):
        cv2.imwrite(self._starting_dir / f"img{self._frame_cntr:05}.jpg", frame)