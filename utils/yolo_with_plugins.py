"""yolo_with_plugins.py

Implementation of TrtYOLO class with the yolo_layer plugins.
"""


from __future__ import print_function

import ctypes

import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda

import queue
import threading
import time

# try:
#     ctypes.cdll.LoadLibrary('./plugins/libyolo_layer.so')
# except OSError as e:
#     raise SystemExit('ERROR: failed to load ./plugins/libyolo_layer.so.  '
#                      'Did you forget to do a "make" in the "./plugins/" '
#                      'subdirectory?') from e

PREALLOC_OUTPUTS = 8

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


def _nms_boxes(detections, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding
    boxes with their confidence scores and return an array with the
    indexes of the bounding boxes we want to keep.

    # Args
        detections: Nx7 numpy arrays of
                    [[x, y, w, h, box_confidence, class_id, class_prob],
                     ......]
    """
    x_coord = detections[:, 0]
    y_coord = detections[:, 1]
    width = detections[:, 2]
    height = detections[:, 3]
    box_confidences = detections[:, 4] * detections[:, 6]

    areas = width * height
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)
        iou = intersection / union
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]

    keep = np.array(keep)
    return keep


def _postprocess_yolo(trt_outputs, category_num, img_w, img_h, conf_th, nms_threshold,
                      input_shape, letter_box=False):
    """Postprocess TensorRT outputs.

    # Args
        trt_outputs: a list of 2 or 3 tensors, where each tensor
                    contains a multiple of 7 float32 numbers in
                    the order of [x, y, w, h, box_confidence, class_id, class_prob]
        conf_th: confidence threshold
        letter_box: boolean, referring to _preprocess_yolo()

    # Returns
        boxes, scores, classes (after NMS)
    """
    # filter low-conf detections and concatenate results of all yolo layers
    # detections = []
    # for o in trt_outputs:
    if True:
        o = trt_outputs
        dets = o.reshape((category_num + 4, -1))
        dets = dets.transpose(1, 0)
        scores = dets[:, 4:].max(axis=1)
        classes = dets[:, 4:].argmax(axis=1)

        dets = dets[:, :4]

        probabilities = np.concatenate(
            (dets, 
            scores.reshape((scores.shape[0], 1)), 
            classes.reshape((scores.shape[0], 1)), 
            np.ones((scores.shape[0], 1))), 
            axis=1)
        # print("Probs ", (probabilities[:, 4] > conf_th).shape)
        probabilities = probabilities[probabilities[:, 4] > conf_th]
        # print("Probs2 ", probabilities.shape)
    detections = probabilities #np.concatenate(probabilities, axis=0)
    # print(detections.shape)

    if len(detections) == 0:
        boxes = np.zeros((0, 4), dtype=np.int)
        scores = np.zeros((0,), dtype=np.float32)
        classes = np.zeros((0,), dtype=np.float32)
    else:
        box_scores = detections[:, 4] * detections[:, 6]

        # scale x, y, w, h from [0, 1] to pixel values
        old_h, old_w = img_h, img_w
        offset_h, offset_w = 0, 0
        if letter_box:
            if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
                old_h = int(input_shape[0] * img_w / input_shape[1])
                offset_h = (old_h - img_h) // 2
            else:
                old_w = int(input_shape[1] * img_h / input_shape[0])
                offset_w = (old_w - img_w) // 2
        detections[:, 0:4] *= np.array(
            [old_w / input_shape[0], old_h / input_shape[1], old_w / input_shape[0], old_h / input_shape[1]], dtype=np.float32)
        detections[:, 0] -= detections[:, 2] / 2
        detections[:, 1] -= detections[:, 3] / 2 

        # NMS
        nms_detections = np.zeros((0, 7), dtype=detections.dtype)
        for class_id in set(detections[:, 5]):
            idxs = np.where(detections[:, 5] == class_id)
            cls_detections = detections[idxs]
            keep = _nms_boxes(cls_detections, nms_threshold)
            nms_detections = np.concatenate(
                [nms_detections, cls_detections[keep]], axis=0)

        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        if letter_box:
            xx = xx - offset_w
            yy = yy - offset_h
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)
        boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1)
        boxes = boxes.astype(np.int)
        scores = nms_detections[:, 4] * nms_detections[:, 6]
        classes = nms_detections[:, 5]
    return boxes, scores, classes


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host        


def get_input_shape(engine):
    """Get input shape of the TensorRT YOLO engine."""
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))


def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    output_idx = 0
    stream = cuda.Stream()
    for binding in engine:
        binding_dims = engine.get_binding_shape(binding)
        if len(binding_dims) == 4:
            # explicit batch case (TensorRT 7+)
            size = trt.volume(binding_dims)
        elif len(binding_dims) == 3:
            # implicit batch case (TensorRT 6 or older)
            size = trt.volume(binding_dims) * engine.max_batch_size
        else:
            raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            assert size % 7 == 0
            outputs.append(HostDeviceMem(host_mem, device_mem))
            output_idx += 1
    assert len(inputs) == 1
    assert len(outputs) == 1
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    """do_inference (for TensorRT 6.x or lower)

    This function is generalized for multiple inputs/outputs.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size,
                          bindings=bindings,
                          stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def do_inference_v2(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class TrtYOLO(object):
    """TrtYOLO class encapsulates things needed to run TRT YOLO."""

    def _load_engine(self):
        TRTbin = 'yolo/%s.trt' % self.model
        with open(TRTbin, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, model, category_num=80, letter_box=False, cuda_ctx=None, conf_th=0.3, visualizer=None, capturer=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.category_num = category_num
        self.letter_box = letter_box
        self.cuda_ctx = cuda_ctx

        self.inference_fn = do_inference if trt.__version__[0] < '7' \
                                         else do_inference_v2
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        self.input_shape = get_input_shape(self.engine)

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

        self.visualizer = visualizer
        self.conf_th = conf_th
        self.stop = False
        self.infer_fps = 0
        self.inference_queue = queue.Queue()
        self.input_queue = queue.Queue()
        # self.input_queue = queue.Queue()
        # self.cuda_thread = threading.Thread(None, self.cuda_infer_fn)
        # self.cuda_thread.start()
        self.postprocess_thread = threading.Thread(None, self.postprocess)
        self.postprocess_thread.start()

        self.cap = capturer
        self.preprocess_thread = threading.Thread(None, self.read_and_preprocess)
        self.preprocess_thread.start()

    def __del__(self):
        """Free CUDA memories."""
        self.stop = True
        self.postprocess_thread.join()
        self.preprocess_thread.join()

        del self.outputs
        del self.inputs
        del self.stream

    def read_and_preprocess(self):
        next_timestamp = 0
        processing_timestamp = 0
        while True:
            start_time = time.time()
            ret, frame = self.cap.read()
            if frame is None:
                self.stop = True
                return

            if processing_timestamp - next_timestamp < 1 / 30:
                """Detect objects in the input image."""
                letter_box = self.letter_box # if letter_box is None else letter_box
                img_resized = _preprocess_yolo(frame, self.input_shape, letter_box)
                self.input_queue.put((frame, img_resized))

            finish_time = time.time()
            # print("Timings ", 1 / 30, finish_time - start_time)
            next_timestamp += 1 / 30
            processing_timestamp = finish_time - start_time
            if 1 / 30 > (finish_time - start_time):
                time.sleep(1 / 30 - (finish_time - start_time))

    def detect(self):
        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.infer_fps += 1
        active_len = self.input_queue.qsize()
        for _ in range(active_len - 1):
            self.input_queue.get()

        if active_len == 0 and self.stop:
            return

        img, img_resized = self.input_queue.get()
        self.inputs[0].host = np.ascontiguousarray(img_resized)
        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        self.infer_fps += 1
        self.inference_queue.put((img, trt_outputs[0].copy()))

        # if not self.result_queue.empty():
        #     frame = self.result_queue.get()
        #     cv2.imshow("Detection", frame)
        #     cv2.waitKey


    def postprocess(self):
        ctr = 0
        window_name = 'signs_detection'

        while True:
            if ctr % 100 == 0:
                print(ctr)
            ctr += 1
            print('.', end='', flush=True)

            if self.inference_queue.qsize() > PREALLOC_OUTPUTS - 2:
                self.inference_queue.get()
                continue

            if self.inference_queue.empty() and self.stop:
                print("Done postprocessing")
                cv2.destroyAllWindows() # destroy all windows
                return

            frame, trt_outputs = self.inference_queue.get()
            boxes, scores, classes = _postprocess_yolo(
                trt_outputs, self.category_num, frame.shape[1], frame.shape[0], self.conf_th,
                nms_threshold=0.5, input_shape=self.input_shape,
                letter_box=self.letter_box)

            # clip x1, y1, x2, y2 within original image
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame.shape[1]-1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame.shape[0]-1)

            frame = self.visualizer.draw_bboxes(frame, boxes, scores, classes)
            # cv2.imwrite(f"/home/artint/images_out/{ctr:05}.jpg", frame)
            cv2.imwrite(f"/home/artint/images_out/{ctr:05}.jpg", frame)
            # cv2.imshow(window_name, frame)
            # cv2.waitKey(1)
            # self.result_queue.put(frame)
            # writer.write(frame)
        # return boxes, scores, classes
