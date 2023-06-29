import numpy as np
import cv2

class LetterBox:
    def __init__(self, target_img_size):
        self._target_img_size = target_img_size

    def add_letterbox(self, resized):
        img_h, img_w, _ = resized.shape
        new_w, new_h = self._target_img_size
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            ratio = new_w / img_w
            new_h = int(img_h * ratio)
            offset_h = (self._target_img_size[1] - new_h) // 2
        else:
            ratio = new_h / img_h
            new_w = int(img_w * ratio)
            offset_w = (self._target_img_size[0] - new_w) // 2
        img = np.full((self._target_img_size[1], self._target_img_size[0], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
        return img


class OpenCVInferencePreparer:
    def __init__(self):
        pass

    def prepare_for_infer(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)
        img /= 255.0
        return img

class OpenCVCapturer:
    def __init__(self, path, resized_size):
        self._path = path
        self._opencv_cap = cv2.VideoCapture(path)
        self._resized_size = resized_size

    def read(self):
        _, frame = self._opencv_cap.read()
        if 1:
            resized_frame = cv2.resize(frame, (self._resized_size[0], self._resized_size[1]));
            return (True, (frame, resized_frame))
        else:
            return False, (None, None)

class OpenCVPreprocessor:
    def __init__(self, strided_img_size):
        self._letterboxer = LetterBox(strided_img_size)
        self._inference_preparer = OpenCVInferencePreparer()

    def transform(self, img):
        img = self._letterboxer.add_letterbox(img)
        img = self._inference_preparer.prepare_for_infer(img)
        return np.ascontiguousarray(img)
