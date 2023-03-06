import sys, os, time, threading
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject

# full_pipeline = "filesrc location=\"NO20230128-115104-009260F.MP4\" ! qtdemux ! queue name=read_queue ! h265parse ! libde265dec ! tee name=decoded ! queue ! videoconvert ! video/x-raw,format=BGR ! appsink name=a1 decoded. ! queue ! videoscale ! video/x-raw,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! appsink"
full_pipeline = "filesrc location=\"NO20230128-115104-009260F.MP4\" ! qtdemux ! queue name=read_queue ! h265parse ! libde265dec ! tee name=decoded ! queue ! videoconvert ! video/x-raw,format=BGR ! jpegenc ! multifilesink location=full_res/fimg%04d.jpg decoded. ! queue ! videoscale ! video/x-raw,width=640,height=480 ! videoconvert ! video/x-raw,format=BGR ! jpegenc ! multifilesink location=small_res/simg%04d.jpg"


class CLI_Main(object):

    def __init__(self):
        self.my_pipeline = Gst.Pipeline.new("my-pipeline")
        # self.player = Gst.ElementFactory.make("playbin", "player")
        self.player = Gst.ElementFactory.make("filesrc", "player")
        fakesink = Gst.ElementFactory.make("multifilesink", "fakesink")
        self.encoder = Gst.ElementFactory.make("jpegenc", "video-output")
        fakesink.set_property("location", "bibibi%d.jpg")
        self.player.set_property("location", sys.argv[1])
        demuxer = Gst.ElementFactory.make("qtdemux", "demuxer")
        demuxer.connect("pad-added", self.demuxer_callback)
        self.fakesink1 = Gst.ElementFactory.make("fakesink", "devzero1")
        # self.player.set_property("video-sink", fakesink)
        self.player.set_property("num-buffers", 100)
        self.my_pipeline.add(self.player)
        self.my_pipeline.add(fakesink)
        self.my_pipeline.add(self.encoder)
        self.my_pipeline.add(demuxer)
        self.my_pipeline.add(self.fakesink1)
        self.player.link(demuxer)
        self.encoder.link(fakesink)
        bus = self.my_pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self.on_message)

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.my_pipeline.set_state(Gst.State.NULL)
            self.playmode = False
        elif t == Gst.MessageType.ERROR:
            self.my_pipeline.set_state(Gst.State.NULL)
            err, debug = message.parse_error()
            print("Error: %s" % err, debug)
            self.playmode = False


    def demuxer_callback(self, demuxer, pad):
        print("Got pad ", pad.get_property("template").name_template, pad.get_property("name"))
        if pad.get_property("template").name_template == "video_%u" and pad.get_property("name") == "video_0":
            # print(dir(self.encoder))
            qv_pad = self.encoder.get_static_pad("sink")
            # print(qv_pad)
            pad.link(qv_pad)
            print("Link 0 finish")
        elif pad.get_property("name") == "audio_0" or pad.get_property("name") == "video_1":
            # print(f"Got {pad.get_property("name")}")
            fakepad = self.fakesink1.get_static_pad("sink")
            # fakepad = Gst.Pad.new(f"sink{self.fakesink1.numpads}", Gst.PadDirection.SINK)
            # fakepad.set_active(True)
            # self.fakesink1.add_pad(fakepad)
            pad.link(fakepad)
            print("Link other finish")

    def start(self):
        print("start")
        for filepath in sys.argv[1:]:
            if os.path.isfile(filepath):
                filepath = os.path.realpath(filepath)
                self.playmode = True
                # self.player.set_property("uri", "file://" + filepath)
                self.my_pipeline.set_state(Gst.State.PLAYING)
                while self.playmode:
                    print(".", end="")
                    time.sleep(1)
        time.sleep(1)
        loop.quit()

def pipeline_loop(pipeline, is_playing):
    print("Running pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    while is_playing[0]:
        print(".", end="")
        time.sleep(1)


GObject.threads_init()
Gst.init(None)    

gst_pipeline = Gst.parse_launch(full_pipeline)
# mainclass = CLI_Main()
# threading.start_new_thread(mainclass.start, ())
# starter = threading.Thread(None, mainclass.start)
is_playing = [True]
starter = threading.Thread(None, pipeline_loop, args=(gst_pipeline, is_playing))
starter.start()
loop = GLib.MainLoop()
loop.run()