import sys, os, time, threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

# full_pipeline = "filesrc location=\"NO20230128-115104-009260F.MP4\" ! qtdemux ! queue name=read_queue ! h265parse ! libde265dec ! tee name=decoded ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink name=a1 decoded. ! queue ! videoscale ! video/x-raw,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"
full_pipeline = "filesrc location=\"NO20230128-115104-009260F.MP4\" ! qtdemux ! queue name=read_queue ! h265parse ! libde265dec ! tee name=decoded ! queue ! videoconvert ! video/x-raw,format=BGR ! jpegenc ! multifilesink location=full_res/fimg%04d.jpg decoded. ! queue ! videoscale ! video/x-raw,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! jpegenc ! multifilesink location=small_res/simg%04d.jpg"

def pipeline_loop(pipeline, is_playing):
    print("Running pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    while is_playing[0]:
        print(".", end="")
        time.sleep(1)


GObject.threads_init()
Gst.init(None)    

gst_pipeline = Gst.parse_launch(full_pipeline)
is_playing = [True]
starter = threading.Thread(None, pipeline_loop, args=(gst_pipeline, is_playing))
starter.start()
loop = GLib.MainLoop()
loop.run()