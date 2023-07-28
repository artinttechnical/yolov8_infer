import time, threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
from collections import namedtuple
from queue import Queue
import numpy as np
import cv2
import statistics

PIPELINE_STR_DESCRIPTION = """
filesrc location={filesrc} ! 
qtdemux ! 
queue name=read_queue ! 
h265parse ! libde265dec ! 
tee name=decoded ! 
queue ! 
videoconvert ! video/x-raw,format=BGR ! 
appsink name={fullres_sink_name}
"""

# decoded. ! 
# queue ! 
# videoscale ! video/x-raw,width={width},height={height} ! 
# videoconvert ! video/x-raw,format=BGR ! 
# appsink name={smallres_sink_name}"""

SinkNames = namedtuple("SinkNames", ["fullres"])
SINK_NAMES = SinkNames("fullres_sink")

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
        self._resolutions = {}
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
            # width=resized_width,
            # height=resized_height,
            # smallres_sink_name=SINK_NAMES.smallres
            ))
        self._is_playing = False

    def _on_buffer(self, sink, data) -> Gst.FlowReturn:
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
            with self._data_ready_cv:
                self._data_ready_cv.notify()
        elif t == Gst.MessageType.ERROR:
            print("Got Error")
            self._is_playing[0] = False
            self._gst_pipeline.set_state(Gst.State.NULL)
        elif t == Gst.MessageType.STATE_CHANGED:
            for sink_name in SINK_NAMES:
                pipeline_appsink = self._gst_pipeline.get_by_name(sink_name)
                pad = pipeline_appsink.get_static_pad("sink")
                caps = pad.get_current_caps()
                if caps:
                    struct = caps.get_structure(0)
                    self._resolutions[sink_name] = (struct.get_int("height")[1], struct.get_int("width")[1], 3)


    def start(self):
        def pipeline_loop(pipeline, is_playing, loop):
            print("Running pipeline")
            pipeline.set_state(Gst.State.PLAYING)
            #TODO - make honest synchronization, not via polling
            while is_playing[0]:
                time.sleep(1)
            print("Stopping pipeline")
            loop.quit()
            print("Pipeline stopped")

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
            print("read EOF return - 1")
            return False, ()
        with self._data_ready_cv:
            while self._is_playing[0] and any([q.empty() for q in self._buffers.values()]):
                self._data_ready_cv.wait()

        if not self._is_playing[0]:
            print("read EOF return - 2")
            return False, ()

        result = [
            np.frombuffer(
                self._buffers[queue_name].get(),
                dtype=np.uint8
            ).reshape(
                self._resolutions[queue_name]
            )
        for queue_name in self._buffers]
        return True, tuple(result)

    #TODO - make honest resolution size, opening status and releasing or remove it at all
    def isOpened(self):
        return True

    def release(self):
        pass

class OpenCVReader:
    def __init__(self, path):
        self._path = path
        self._opencv_cap = cv2.VideoCapture(path)

    def start(self):
        pass

    def read(self):
        ret, frame = self._opencv_cap.read()
        if ret:
            return (True, (frame))
        else:
            return False, (None, None)
        
    def stop(self):
        pass



class GstreamerReader:
    def __init__(self, path):
        self._capturer = HarcodedGstreamerPipeline(path, 0, 0) #, 2592x1944)
        self._capturer.start()

    def read(self):
        ret, frame = self._capturer.read()
        if ret:
            resized_frame = cv2.resize(frame[0], (640, 480));
            return (True, (frame[0], resized_frame))
        else:
            return False, (None, None)

    
    def __del__(self):
        self._capturer.stop()

if __name__ == "__main__":
    import cv2

    # capturer = HarcodedGstreamerPipeline("NO20230128-115104-009260F.MP4", 640, 480)
    capturer = OpenCVReader("NO20230128-115104-009260F.MP4")
    capturer.start()
    counter = 0
    times = []
    while True:
        start_t = time.time()
        ret, frames = capturer.read()
        if not ret:
            break
        # fullres_frame, smallres_frame = frames

        end_t = time.time()
        times.append(end_t - start_t)

        frames

        
            
        # cv2.imwrite(f"full_res/fimg{counter:04}.jpg", fullres_frame)
        # cv2.imwrite(f"small_res/simg{counter:04}.jpg", smallres_frame)
        counter += 1

    capturer.stop()
    tm_mean = statistics.mean(times)
    quartiles = statistics.quantiles(times)
    perc = statistics.quantiles(times, n=100)
    print(f"Mean {tm_mean}, deviation {statistics.pstdev(times, tm_mean)}, 25-75 {quartiles[0]} - {quartiles[-1]}, 99-perc {perc[-1]}")
