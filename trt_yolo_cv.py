"""trt_yolo_cv.py

This script could be used to make object detection video with
TensorRT optimized YOLO engine.

"cv" means "create video"
made by BigJoon (ref. jkjung-avt)
"""


import os
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
from utils.gstreamer_pipeline import NvidiaAcceleratedCapturer

from pathlib import Path


import time

def parse_args():
    """Parse input arguments."""
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-v', '--video', type=str, required=True,
        help='input video file name')
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='output video file name')
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument("-n", "--nvidia-accel", action="store_true",
                        help="Use nvidia-accelerated gstreamer video processing")
    args = parser.parse_args()
    return args


def loop_and_detect(cap, trt_yolo, conf_th, writer):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cap: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      writer: the VideoWriter object for the output video.
    """
    ctr = 0
    while not trt_yolo.stop:
        # ret, frame = cap.read()
        # if frame is None:  break
        # boxes, confs, clss = trt_yolo.detect(frame, conf_th)
        trt_yolo.detect()

    print('\nDone.')

class MyCap:
    def __init__(self, path: str):
        self._path = Path(path)
        self._counter = 1200
        # self._counter = 1694
        # self._counter = 1 
    
    def get(self, param):
        if param == 3:
            # return 2592
            return 640 * 2
        elif param == 4:
            return 480 * 2

    def read(self):
        target_file = self._path / f"image009260F_{self._counter:04}.jpg"
        self._counter += 1
        if target_file.exists():
            return True, cv2.imread(str(target_file))
        else:
            return False, None

    def isOpened(self):
        return True

    def release(self):
        pass

def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cap = NvidiaAcceleratedCapturer(args.video, 640, 480)

    if not cap.isOpened():
        raise SystemExit('ERROR: failed to open the input video file!')
    # frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    writer=None
    # writer = cv2.VideoWriter(
    #     args.output,
    #     cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    if args.category_num == 155:
        prefix = "sign_images/RU_road_sign_"
        suffix = ".svg.png"
        file_name = "labels.txt"
    elif args.category_num == 4:
        prefix = "traffic_lights/traffic-light-"
        suffix = ".jpg"
        file_name = "labels_lights.txt"

    cls_dict = get_cls_dict(args.category_num, file_name)

    vis = BBoxVisualization(cls_dict, prefix, suffix)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box, conf_th=0.3, visualizer=vis, capturer=cap)

    total_start_time = time.time()
    loop_and_detect(cap, trt_yolo, conf_th=0.3, writer=writer)
    total_end_time = time.time()

    print("Total FPS ", 1800 / (total_end_time - total_start_time))
    print("Infer FPS ", trt_yolo.infer_fps / (total_end_time - total_start_time))
    # print("Output FPS ", 600 / (total_end_time - total_start_time))

    # writer.release()
    cap.release()


if __name__ == '__main__':
    main()
