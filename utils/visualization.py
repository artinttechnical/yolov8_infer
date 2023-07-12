"""visualization.py

The BBoxVisualization class implements drawing of nice looking
bounding boxes based on object detection results.
"""


import numpy as np
import cv2


# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: total number of colors/classes.

    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import random
    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
    cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1]:topleft[1]+h, topleft[0]:topleft[0]+w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class BBoxWithImagesVisualization():
    """BBoxVisualization class implements nice drawing of boudning boxes.

    # Arguments
      cls_dict: a dictionary used to translate class id to its name.
    """

    def __init__(self, classes_container):
        self._classes_container = classes_container
        self._colors = gen_colors(classes_container.get_classes_num())

    def _calculate_shape(self, sign_images, img):
        widths = 0
        height = -1
        ORIG_IMAGE_FRACTION = 8

        for sign_image in sign_images.values():
            ratio = sign_image.shape[1] / (img.shape[1] / ORIG_IMAGE_FRACTION)
            sign_images.append(
              cv2.resize(
                sign_image, 
                (
                  int(sign_image.shape[1] / ratio), 
                  int(sign_image.shape[0] / ratio))))
            height = max(height, sign_images[-1].shape[0])
            widths += sign_images[-1].shape[1] + 20

        widths -= 20
        return widths, height
    
    def _calculate_starting_positions(self, widths, height, img):
        cur_sign_x = img.shape[1] / 2 - widths / 2
        if cur_sign_x < 0:
          cur_sign_x = 0

        sign_top = img.shape[0] - height - 10
        return cur_sign_x, sign_top

    def draw_bboxes(self, img, boxes, confs, classes):
        """Draw detected bounding boxes on the original image."""
        sign_images = []
        

        sign_images = self._classes_container.get_images_for_classes(classes)
        signs_width, signs_height = self._calculate_shape(sign_images, img)
        cur_sign_x, sign_top = self._calculate_starting_positions(img, signs_width, signs_height, img)

        # print("Len ", len(confs))
        updated_image = img
        for bbox, det_class in zip(boxes, classes):
            updated_image = self._put_bounding_box(updated_image, bbox, self._class_names[det_class], self._colors[det_class])
            sign_image = sign_images[det_class]
            updated_image = self._put_sign_image_and_line(updated_image, sign_image, bbox, cur_sign_x, sign_top, self._colors[det_class])
            cur_sign_x += sign_image.shape[1] + 20


        return updated_image

    def _put_bounding_box(self, img, bb, clas_name, color):
          x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
          cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

    def _put_sign_image_and_line(img, sign_image, bb, cur_sign_x, sign_top, color):
        x_min, _, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
        line_src = x_max if x_max < img.shape[1] / 2 else x_min

        cv2.line(img, (line_src, y_max), (int(cur_sign_x + sign_image.shape[1] / 2), sign_top), color, 2)
        cv2.rectangle(
          img, 
          (int(cur_sign_x) - 2, sign_top - 2), 
          (int(cur_sign_x) + sign_image.shape[1] + 2, sign_top + sign_image.shape[0] + 2), 
          color, 2)

        img[
              sign_top:sign_top + sign_image.shape[0], 
              int(cur_sign_x):int(cur_sign_x) + sign_image.shape[1]] = sign_image
        return img
