"""
This script could be used to make object detection video with
TensorRT optimized YOLO engine.

"cv" means "create video"
made by BigJoon (ref. jkjung-avt)
"""


import os
import argparse

from yolo_detector import YoloDetector
from yolo_preprocessor import OpenCVCapturer, OpenCVPreprocessor
from ultralytics_inferer import UltralyticsInferer
from yolo_postprocessor import YoloPostprocessor
from visualizers import OpenCVImagesVisualizer, OpencvJpegStorer
from text_file_names_images_in_folder import TextFileNamesImagesInFolder


def parse_args():
    """Parse input arguments."""
    desc = ('Run the TensorRT optimized object detecion model on an input '
            'video and save BBoxed overlaid output as another video.')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '-v', '--video', type=str, required=True,
        help='input video file name')
    parser.add_argument(
        '-o', '--output', type=str, required=False,
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
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    classes_data = TextFileNamesImagesInFolder("", "")
    capturer = OpenCVCapturer(
        "/home/artint/Projects/MachineLearning/Otus2022/Project/datasets/MyRegDataset/NO20230128-115104-009260F.MP4", 
        (640, 480))
    img_transformer = OpenCVPreprocessor((640, 480))
    inferer = UltralyticsInferer("/home/artint/Projects/MachineLearning/Otus2022/Project/models/signs_best_small.pt")
    postprocessor = YoloPostprocessor(
        (2592 / 640, 1944 / 480), classes_data.get_classes_num(), 0.3, 0.5)
    
    visualizer = OpenCVImagesVisualizer(classes_data)
    storer = OpencvJpegStorer("output")
    detector = YoloDetector(
        capturer,
        img_transformer,
        inferer,
        postprocessor,
        visualizer,
        storer,
        realtime=False)

    detector.process()


if __name__ == '__main__':
    main()
