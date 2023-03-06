# import gi
# gi.require_version('Gst', '1.0')

# from gi.repository import Gst, GLib, GObject
# import threading
# import time


# GObject.threads_init()
# Gst.init(None)   

# cv2_adapter = Gst.Pipeline.new("Adapter")

# player = Gst.Pipeline.new("player")
# source = Gst.ElementFactory.make("videotestsrc", "video-source")
# source.set_property("num-buffers", 1)
# sink = Gst.ElementFactory.make("jpegenc", "video-output")
# sink1 = Gst.ElementFactory.make("filesink", "video-output1")
# sink1.set_property("location", "full_res/img%d.jpg")
# caps = Gst.Caps.from_string("video/x-raw, width=320, height=230")
# filter = Gst.ElementFactory.make("capsfilter", "filter")
# filter.set_property("caps", caps)
# player.add(source)
# # player.add(filter)
# player.add(sink)
# player.add(sink1)
# source.link(sink)
# # source.link(filter)
# # filter.link(sink)
# sink.link(sink1)


# def on_message(bus, message, is_running, index):
#     t = message.type
#     if t == Gst.MessageType.EOS:
#         is_running[index] = False

# def pipeline_run(pipeline, is_running):
#     pipeline.set_state(Gst.State.PLAYING)
#     while is_running[0] or is_running[1]:
#         print("Wait ")
#         time.sleep(1)
#     time.sleep(1)
#     loop.quit()

# is_running = [True, False]
# bus = player.get_bus()
# bus.add_signal_watch()
# bus.connect("message", on_message, is_running, 0)

# start_thread = threading.Thread(None, pipeline_run, "Starting", args=(cv2_adapter, is_running))
# start_thread.start()
# loop = GLib.MainLoop()
# loop.run()
# start_thread.join()

import sys, os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject, Gtk

# Needed for window.get_xid(), xvimagesink.set_window_handle(), respectively:
from gi.repository import GdkX11, GstVideo

class GTK_Main(object):

    def __init__(self):
        window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
        window.set_title("Video-Player")
        window.set_default_size(500, 400)
        window.connect("destroy", Gtk.main_quit, "WM destroy")
        vbox = Gtk.VBox()
        window.add(vbox)
        hbox = Gtk.HBox()
        vbox.pack_start(hbox, False, False, 0)
        self.entry = Gtk.Entry()
        hbox.add(self.entry)
        self.button = Gtk.Button("Start")
        hbox.pack_start(self.button, False, False, 0)
        self.button.connect("clicked", self.start_stop)
        self.movie_window = Gtk.DrawingArea()
        vbox.add(self.movie_window)
        window.show_all()

        self.player = Gst.ElementFactory.make("playbin", "player")
        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect("message", self.on_message)
        bus.connect("sync-message::element", self.on_sync_message)

    def start_stop(self, w):
        if self.button.get_label() == "Start":
            filepath = self.entry.get_text().strip()
            if os.path.isfile(filepath):
                filepath = os.path.realpath(filepath)
                self.button.set_label("Stop")
                self.player.set_property("uri", "file://" + filepath)
                self.player.set_state(Gst.State.PLAYING)
            else:
                self.player.set_state(Gst.State.NULL)
                self.button.set_label("Start")

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.player.set_state(Gst.State.NULL)
            self.button.set_label("Start")
        elif t == Gst.MessageType.ERROR:
            self.player.set_state(Gst.State.NULL)
            err, debug = message.parse_error()
            print("Error: %s" % err, debug)
            self.button.set_label("Start")

    def on_sync_message(self, bus, message):
        if message.get_structure().get_name() == 'prepare-window-handle':
            imagesink = message.src
            imagesink.set_property("force-aspect-ratio", True)
            imagesink.set_window_handle(self.movie_window.get_property('window').get_xid())


GObject.threads_init()
Gst.init(None)        
GTK_Main()
Gtk.main()