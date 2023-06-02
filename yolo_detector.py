import collections

class YoloDetector:
    def __init__(self, capturer, img_transformer, inferer, postprocessor, visualizer):
        self._capturer = capturer
        self._img_transformer = img_transformer
        self._inferer = inferer
        self._postprocessor = postprocessor
        self._visualizer = visualizer

        self._preprocessing_thread = None
        self._postprocessing_thread = None

        self._global_stop = False

        #10 - from the head
        self._frames_storage = collections.deque(maxlen=10)

    def _preprocess_tf(self):
        pass

    def _postprocess_tf(self):
        pass

    def main_fn(self):
        pass