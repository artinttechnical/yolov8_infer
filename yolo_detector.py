import collections
#TODO - replace to some kind of async
import threading



class YoloDetector:
    def __init__(self, capturer, img_transformer, inferer, postprocessor, visualizer, result_storer):
        self._capturer = capturer
        self._img_transformer = img_transformer
        self._inferer = inferer
        self._postprocessor = postprocessor
        self._visualizer = visualizer
        self._result_storer = result_storer

        self._preprocessing_thread = threading.Thread(None, self._preprocess_tf)
        # self._preprocessing_thread.start()

        self._postprocessing_thread = threading.Thread(None, self._postprocess_tf)
        # self._postprocessing_thread.start()

        self._global_stop = False

        #magic numbers approximately correspond to smth 
        self._infer_queue = collections.deque(maxlen=20)
        self._result_queue = collections.deque(maxlen=10)

    def _preprocess_tf(self):
        while True:
            ret, frames = self._cap.read()
            if not ret:
                self._global_stop = True
                break
            
            orig_img, resized_imd = frames
            ready_for_infer_img = self._img_transformer.transform(resized_imd)
            self._infer_queue.append(orig_img, ready_for_infer_img)


    def _postprocess_tf(self):
        while not self._global_stop:
            orig_frame, raw_infer_results = self._result_queue.pop()
            objects = self._postprocessor.process_raw_data(raw_infer_results)
            resulting_frame = self._visualizer.draw_objects(orig_frame, objects)

            #TODO - estimate probably move to other storage
            self._result_storer.store(resulting_frame)
            
    def _main_fn(self):
        pass