import time, threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
from collections import namedtuple
from queue import Queue
import numpy as np

PIPELINE_STR_DESCRIPTION = """
filesrc location={filesrc} ! 
qtdemux ! 
queue name=read_queue ! 
h265parse ! libde265dec ! 
tee name=decoded ! 
queue ! 
videoconvert ! video/x-raw,format=BGR ! 
appsink name={fullres_sink_name} 
decoded. ! 
queue ! 
videoscale ! video/x-raw,width={width},height={height} ! 
videoconvert ! video/x-raw,format=BGR ! 
appsink name={smallres_sink_name}"""

SinkNames = namedtuple("SinkNames", ["fullres", "smallres"])
SINK_NAMES = SinkNames("fullres_sink", "smallres_sink")

class HarcodedGstreamerPipeline:
    def __init__(self, path, resized_width, resized_height) -> None:
        self._general_gstreamer_init()
        self._create_and_set_pipeline(path, resized_width, resized_height)
        self._subscribe_to_eos_msg()
        self._subscribe_to_appsinks_data()

    def _subscribe_to_eos_msg(self):
        self._is_playing = [False]
        bus = self._gst_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_message)

    def _subscribe_to_appsinks_data(self):
        self._buffers = {}
        self._data_ready_cv = threading.Condition()
        for sink_name in SINK_NAMES:
            self._buffers[sink_name] = Queue()
            pipeline_appsink = self._gst_pipeline.get_by_name(sink_name)
            # print("Appsink ", pipeline_appsink)
            pipeline_appsink.set_property("emit-signals", True)
            pipeline_appsink.connect("new-sample", self._on_buffer, None)

    # @staticmethod
    def _general_gstreamer_init(self):
        GObject.threads_init()
        Gst.init(None)

    def _create_and_set_pipeline(self, path, resized_width, resized_height):
        self._gst_pipeline = Gst.parse_launch(
            PIPELINE_STR_DESCRIPTION.format(
            filesrc=path, 
            fullres_sink_name=SINK_NAMES.fullres,
            width=resized_width,
            height=resized_height,
            smallres_sink_name=SINK_NAMES.smallres))
        self._is_playing = False

    def _on_buffer(self, sink, data) -> Gst.FlowReturn:
        # print("On buffer")
        sink_name = sink.get_property("name")
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        # print(type(buffer))
        with self._data_ready_cv:
            self._buffers[sink_name].put(buffer.extract_dup(0, buffer.get_size()))
            self._data_ready_cv.notify()

        return Gst.FlowReturn.OK
    
    def _on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            print("Got EOS")
            self._is_playing[0] = False
            self._gst_pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.ERROR:
            print("Got Error")
            self._is_playing[0] = False
            self._gst_pipeline.set_state(Gst.State.NULL)

    def start(self):
        def pipeline_loop(pipeline, is_playing, loop):
            print("Running pipeline")
            pipeline.set_state(Gst.State.PLAYING)
            #TODO - make honest synchronization, not via polling
            while is_playing[0]:
                time.sleep(1)
            print("Stopping pipeline")
            loop.quit()

        loop = GLib.MainLoop()
        self._pipeline_thread = threading.Thread(None, pipeline_loop, args=(self._gst_pipeline, self._is_playing, loop))
        self._is_playing[0] = True
        self._pipeline_thread.start()
        self._main_loop_thread = threading.Thread(None, lambda : loop.run())
        self._main_loop_thread.start()

    def stop(self):
        self._pipeline_thread.join()
        self._main_loop_thread.join()

    #interface part
    def read(self):
        if not self._is_playing[0]:
            return False, ()
        with self._data_ready_cv:
            while any([q.empty() for q in self._buffers.values()]):
                self._data_ready_cv.wait()
        result = [queue.get() for queue in self._buffers.values()]
        #TODO - replace hardcoded resolution
        result[0] = np.frombuffer(result[0], dtype=np.uint8)
        result[1] = np.frombuffer(result[1], dtype=np.uint8)
        result[0] = result[0].reshape((1944, 2592, 3))
        result[1] = result[1].reshape((480, 640, 3))
        return True, tuple(result)

    #TODO - make honest resolution size, opening status and releasing or remove it at all
    def isOpened(self):
        return True

    def release(self):
        pass


if __name__ == "__main__":
    import cv2

    capturer = HarcodedGstreamerPipeline("NO20230128-115104-009260F.MP4", 640, 480)
    capturer.start()
    counter = 0
    while True:
        ret, frames = capturer.read()
        if not ret:
            break
        fullres_frame, smallres_frame = frames
        
        cv2.imwrite(f"full_res/fimg{counter:04}.jpg", fullres_frame)
        cv2.imwrite(f"small_res/simg{counter:04}.jpg", smallres_frame)
        counter += 1

    capturer.stop()
