import gi
gi.require_version('Gst', '1.0')

from gi.repository import Gst, GLib, GObject
import threading
import time


GObject.threads_init()
Gst.init(None)   

cv2_adapter = Gst.Pipeline.new("Adapter")

source = Gst.ElementFactory.make("filesrc", "video-source")
source.set_property("location", "NO20230128-115104-009260F.MP4")
cv2_adapter.add(source)

demuxer = Gst.ElementFactory.make("qtdemux", "demuxer")
cv2_adapter.add(demuxer)

videoqueue = Gst.ElementFactory.make("queue", "videoqueue")
cv2_adapter.add(videoqueue)

parser = Gst.ElementFactory.make("h265parse", "stream-parser")
cv2_adapter.add(parser)

decoder = Gst.ElementFactory.make("libde265dec", "stream-decoder")
cv2_adapter.add(decoder)

streams_tee = Gst.ElementFactory.make("tee", "streams-tee")
cv2_adapter.add(streams_tee)

fullres_queue = Gst.ElementFactory.make("queue", "full-resolution-queue")
cv2_adapter.add(fullres_queue)

fullres_format_converter = Gst.ElementFactory.make("videoconvert", "fullres_format_converter")
cv2_adapter.add(fullres_format_converter)

fullres_format_caps = Gst.Caps.from_string("video/x-raw,format=BGR")
fullres_format_filter = Gst.ElementFactory.make("capsfilter", "fullres_format_filter")
fullres_format_filter.set_property("caps", fullres_format_caps)
cv2_adapter.add(fullres_format_filter)

fullres_encoder = Gst.ElementFactory.make("jpegenc", "full-resolution-encoder")
cv2_adapter.add(fullres_encoder)

fullres_sink = Gst.ElementFactory.make("multifilesink", "full-resolution-sink")
fullres_sink.set_property("location", "full_res/img%05d.jpg")
cv2_adapter.add(fullres_sink)

resized_queue = Gst.ElementFactory.make("queue", "resized-queue")
cv2_adapter.add(resized_queue)
resized_scaler = Gst.ElementFactory.make("videoscale")
cv2_adapter.add(resized_scaler)

resize_caps = Gst.Caps.from_string("video/x-raw, width=640, height=480")
resize_filter = Gst.ElementFactory.make("capsfilter", "resize_filter")
resize_filter.set_property("caps", resize_caps)
cv2_adapter.add(resize_filter)

resized_format_converter = Gst.ElementFactory.make("videoconvert", "resized_format_converter")
cv2_adapter.add(resized_format_converter)

resized_format_caps = Gst.Caps.from_string("video/x-raw,format=BGR")
resized_format_filter = Gst.ElementFactory.make("capsfilter", "resized_format_filter")
resized_format_filter.set_property("caps", resized_format_caps)
cv2_adapter.add(resized_format_filter)

resized_encoder = Gst.ElementFactory.make("jpegenc", "resized-encoder")
cv2_adapter.add(resized_encoder)

resized_sink = Gst.ElementFactory.make("multifilesink", "resized-sink")
resized_sink.set_property("location", "small_res/img%05d.jpg")
cv2_adapter.add(resized_sink)

source.link(demuxer)
demuxer.link(videoqueue)
videoqueue.link(parser)
parser.link(decoder)
decoder.link(streams_tee)

# queue_sinkpad = fullres_queue.get_pad("sink")
# tee_audio_pad = gst_element_request_pad_simple (tee, "src_%u");
# streams_tee_pad.link(queue_sinkpad)
streams_tee.link(fullres_queue)

fullres_queue.link(fullres_format_converter)
fullres_format_converter.link(fullres_format_filter)
fullres_format_filter.link(fullres_encoder)
fullres_encoder.link(fullres_sink)


# streams_tee.link(resized_queue)
# resized_queue.link(resized_scaler)
# resized_scaler.link(resize_filter)
# resize_filter.link(resized_format_converter)
# resized_format_converter.link(resized_format_filter)
# resized_format_filter.link(resized_encoder)
# resized_encoder.link(resized_sink)

def on_message(bus, message, index):
    t = message.type
    if t == Gst.MessageType.EOS:
        is_running = False

def pipeline_run(pipeline, is_running):
    pipeline.set_state(Gst.State.PLAYING)
    while is_running[0] or is_running[1]:
        print("Wait ")
        time.sleep(1)
    time.sleep(1)
    loop.quit()


bus = cv2_adapter.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message, 0)

is_running = [True, False]   
start_thread = threading.Thread(None, pipeline_run, "Starting", args=(cv2_adapter, is_running))
start_thread.start()
loop = GLib.MainLoop()
loop.run()
start_thread.join()