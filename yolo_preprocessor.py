import numpy as np
import cv2

import time

def _preprocess_yolo(img, input_shape, letter_box=False):
    """Preprocess an image before TRT YOLO inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)
        letter_box: boolean, specifies whether to keep aspect ratio and
                    create a "letterboxed" image for inference

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    if letter_box:
        img_h, img_w, _ = img.shape
        new_h, new_w = input_shape[0], input_shape[1]
        offset_h, offset_w = 0, 0
        if (new_w / img_w) <= (new_h / img_h):
            ratio = new_w / img_w
            new_h = int(img_h * ratio)
            offset_h = (input_shape[0] - new_h) // 2
        else:
            ratio = new_h / img_h
            new_w = int(img_w * ratio)
            offset_w = (input_shape[1] - new_w) // 2
        resized = cv2.resize(img, (new_w, new_h))
        img = np.full((input_shape[0], input_shape[1], 3), 127, dtype=np.uint8)
        img[offset_h:(offset_h + new_h), offset_w:(offset_w + new_w), :] = resized
    else:
        img = cv2.resize(img, (input_shape[1], input_shape[0]))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1)).astype(np.float32)
    img /= 255.0
    return img

    def read_and_preprocess(self):
        next_timestamp = 0
        processing_timestamp = 0
        while True:
            start_time = time.time()
            ret, (img_orig, img_resized) = self.cap.read()
            if not ret:
                self.stop = True
                return

            if processing_timestamp - next_timestamp < 1 / 30:
                """Detect objects in the input image."""
                letter_box = self.letter_box # if letter_box is None else letter_box
                img_resized = _preprocess_yolo(img_resized, self.input_shape, letter_box)
                self.input_queue.put((img_orig, img_resized))

            finish_time = time.time()
            # print("Timings ", 1 / 30, finish_time - start_time)
            next_timestamp += 1 / 30
            processing_timestamp = finish_time - start_time
            if 1 / 30 > (finish_time - start_time):
                time.sleep(1 / 30 - (finish_time - start_time))