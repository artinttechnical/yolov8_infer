import numpy as np
import cv2    

OBJECT_COORDINATES_NUM = 4 #x, y, w, h

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
    def __init__(self, axis_scaling, categories_num, object_probability, nms_threshold):
        self._axis_scaling = axis_scaling
        self._categories_num = categories_num
        self._object_probability = object_probability
        self._nms_threshold = nms_threshold

    def process_raw_data(self, infer_results):
        yoloscaled_detections = self._get_detections(infer_results)
        if len(yoloscaled_detections) == 0:
            return self._generate_empty_detections()
        else:
            fullscaled_detections = self._rescale_detections(yoloscaled_detections)
            filtered_detections = self._nms_filter_detections(fullscaled_detections)
            screen_clipped_detections = filtered_detections #self._clip_to_screen(filtered_detections)
            return screen_clipped_detections

    def _get_detections(self, infer_results):
        o = infer_results
        dets = o.reshape((self._categories_num + OBJECT_COORDINATES_NUM, -1))
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
        detections = probabilities[probabilities[:, 4] > self._object_probability]

        return detections

    def _rescale_detections(self, yoloscaled_detections):
        # if letter_box:
        rescaled_detections = yoloscaled_detections
        if False:
            if (img_w / input_shape[1]) >= (img_h / input_shape[0]):
                old_h = int(input_shape[0] * img_w / input_shape[1])
                offset_h = (old_h - img_h) // 2
            else:
                old_w = int(input_shape[1] * img_h / input_shape[0])
                offset_w = (old_w - img_w) // 2
        rescaled_detections[:, 0:4] *= np.array(
            [self._axis_scaling[0], self._axis_scaling[1], self._axis_scaling[0], self._axis_scaling[1]], dtype=np.float32)
        #Yolov8 predicts center of rectangle
        rescaled_detections[:, 0] -= rescaled_detections[:, 2] / 2
        rescaled_detections[:, 1] -= rescaled_detections[:, 3] / 2 
        return rescaled_detections

    def _nms_filter_detections(self, detections):
            # NMS
        nms_detections = np.zeros((0, 7), dtype=detections.dtype)
        for class_id in set(detections[:, 5]):
            idxs = np.where(detections[:, 5] == class_id)
            cls_detections = detections[idxs]
            keep = _nms_boxes(cls_detections, self._nms_threshold)
            nms_detections = np.concatenate(
                [nms_detections, cls_detections[keep]], axis=0)

        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        # if letter_box:
        #     xx = xx - offset_w
        #     yy = yy - offset_h
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
