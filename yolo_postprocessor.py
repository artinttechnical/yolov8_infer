import numpy as np
import cv2    

def _nms_boxes(self, detections, nms_threshold):
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


class YoloPostprocessor:
    def __init__(self, orig_image_shape, categories_num, object_probability, nms_threshold):
        self._orig_image_shape = orig_image_shape
        self._categories_num = categories_num
        self._object_probability = object_probability
        self._nms_threshold = nms_threshold

    def process_raw_data(self, infer_results):
        yoloscaled_detections = self._get_detections(infer_results)
        fullscaled_detections = self._rescale_detections(yoloscaled_detections)
        filtered_detections = self._nms_filter_detections(fullscaled_detections)
        screen_clipped_detections = self._clip_to_screen(filtered_detections)
        return screen_clipped_detections

    def _get_detections(self, infer_results):
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

    def _rescale_detections(self, yoloscaled_detections):
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

    def _nms_filter_detections(self, detections):
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


    def _clip_to_screen(self, boxes):
        if 1:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame.shape[1]-1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame.shape[0]-1)
        return boxes
