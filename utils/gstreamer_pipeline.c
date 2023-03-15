#include <gst/gst.h>
#include <stdio.h>

char PIPELINE_STR_DESCRIPTION[] = ""
"filesrc location=%s ! "
"qtdemux ! "
"queue name=read_queue ! "
"h265parse ! libde265dec ! "
"tee name=decoded ! "
"queue ! "
"videoconvert ! video/x-raw,format=BGR ! "
"appsink name=%s ";
// "decoded. ! "
// "queue ! "
// "videoscale ! video/x-raw,width=%d,height=%d ! "
// "videoconvert ! video/x-raw,format=BGR ! "
// "appsink name=%s";

char FULLRES_SINK_NAME[] = "fullres_sink";
char SMALLRES_SINK_NAME[] = "smallres_sink";

static GstFlowReturn new_sample (GstElement *sink, void* dummy) {
    GstSample *sample;

    /* Retrieve the buffer */
    g_signal_emit_by_name (sink, "pull-sample", &sample);
    if (sample) {
        /* The only thing we do in this example is print a * to indicate a received buffer */
        g_print ("*");
        gst_sample_unref (sample);
        return GST_FLOW_OK;
    }

    return GST_FLOW_ERROR;
}

/* This function is called when an error message is posted on the bus */
static void error_cb (GstBus *bus, GstMessage *msg, GMainLoop * main_loop) {
    GError *err;
    gchar *debug_info;

    /* Print error details on the screen */
    gst_message_parse_error (msg, &err, &debug_info);
    g_printerr ("Error received from element %s: %s\n", GST_OBJECT_NAME (msg->src), err->message);
    g_printerr ("Debugging information: %s\n", debug_info ? debug_info : "none");
    g_clear_error (&err);
    g_free (debug_info);

    g_main_loop_quit (main_loop);
}

/* This function is called when an error message is posted on the bus */
static void eos_cb (GstBus *bus, GstMessage *msg, GMainLoop * main_loop) {
    g_print ("End-Of-Stream reached.\n");
    g_main_loop_quit (main_loop);
}


int
tutorial_main (int argc, char *argv[])
{
    // CustomData data;
    GMainLoop *main_loop;
    GstElement *pipeline, *fullres_appsink, *smallres_appsink;
    GstBus *bus;
    GstMessage *msg;

    /* Initialize GStreamer */
    gst_init (&argc, &argv);

    char pipeline_target[4096];
    snprintf(pipeline_target, 4096, PIPELINE_STR_DESCRIPTION,
    "NO20230128-115104-009260F.MP4", 
    FULLRES_SINK_NAME //, 
    // 640, 480, SMALLRES_SINK_NAME
    );

    printf("%s\n", pipeline_target);
    /* Build the pipeline */
    pipeline =
        gst_parse_launch
        (pipeline_target,
        NULL);

    fullres_appsink = gst_bin_get_by_name(GST_BIN (pipeline), FULLRES_SINK_NAME);
    g_object_set (fullres_appsink, "emit-signals", TRUE, NULL);
    g_signal_connect (fullres_appsink, "new-sample", G_CALLBACK (new_sample), NULL);
    gst_object_unref(fullres_appsink);

    /* Start playing */
    gst_element_set_state (pipeline, GST_STATE_PLAYING);
    /* Create a GLib Main Loop and set it to run */
    main_loop = g_main_loop_new (NULL, FALSE);

    /* Wait until error or EOS */
    bus = gst_element_get_bus (pipeline);
    gst_bus_add_signal_watch (bus);
    g_signal_connect (G_OBJECT (bus), "message::error", (GCallback)error_cb, main_loop);
    g_signal_connect (G_OBJECT (bus), "message::eos", (GCallback)eos_cb, main_loop);
    gst_object_unref (bus);

    g_main_loop_run (main_loop);


    /* Free resources */
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (pipeline);

    return 0;
}

int
main (int argc, char *argv[])
{
    return tutorial_main (argc, argv);
}

