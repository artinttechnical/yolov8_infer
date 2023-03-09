import time, threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

pipeline_str_description = """
filesrc location=\"NO20230128-115104-009260F.MP4\" ! 
qtdemux ! 
queue name=read_queue ! 
h265parse ! libde265dec ! 
tee name=decoded ! 
queue ! 
videoconvert ! video/x-raw,format=BGR ! 
jpegenc ! appsink name=fullres_sink 
decoded. ! 
queue ! 
videoscale ! video/x-raw,width=640,height=480 ! 
videoconvert ! video/x-raw,format=BGR ! 
jpegenc ! appsink name=smallres_sink"""

def pipeline_loop(pipeline, is_playing, loop):
    print("Running pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    while is_playing[0]:
        print(".", end="")
        time.sleep(1)
    print("Stopping pipeline")
    loop.quit()


def on_buffer(sink, data) -> Gst.FlowReturn:
    # print("On buffer")
    global counter
    sink_name = sink.get_property("name")
    sample = sink.emit("pull-sample")
    buffer = sample.get_buffer()
    # print(type(buffer))
    # print("Sink name got ", sink_name, type(data))
    if sink_name == "fullres_sink":
        with open(f"full_res/fimg{counter:04}.jpg", "wb") as f:
            f.write(buffer.extract_dup(0, buffer.get_size()))
    else:
        with open(f"small_res/simg{counter:04}.jpg", "wb") as f:
            f.write(buffer.extract_dup(0, buffer.get_size()))
    counter += 1
    return Gst.FlowReturn.OK

def on_message(bus, message, is_playing, pipeline):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("Got EOS")
        is_playing[0] = False
        pipeline.set_state(Gst.State.NULL)


GObject.threads_init()
Gst.init(None)

gst_pipeline = Gst.parse_launch(pipeline_str_description)
is_playing = [True]
bus = gst_pipeline.get_bus()
bus.add_signal_watch()
bus.connect("message", on_message, is_playing, gst_pipeline)

for sink_name in ("fullres_sink", "smallres_sink"):
    pipeline_appsink = gst_pipeline.get_by_name(sink_name)
    print("Appsink ", pipeline_appsink)
    pipeline_appsink.set_property("emit-signals", True)
    pipeline_appsink.connect("new-sample", on_buffer, None)

# smallres_appsink = gst_pipeline.get_by_name("smallres_sink")
# smallres_appsink.connect("new-sample", on_buffer, None)

counter = 0
loop = GLib.MainLoop()
starter = threading.Thread(None, pipeline_loop, args=(gst_pipeline, is_playing, loop))
starter.start()
loop.run()
print("Quit loop")
starter.join()
print("Thread joined")
