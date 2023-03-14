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
"appsink name=%s "
"decoded. ! "
"queue ! "
"videoscale ! video/x-raw,width=%d,height=%d ! "
"videoconvert ! video/x-raw,format=BGR ! "
"appsink name=%s";


int
tutorial_main (int argc, char *argv[])
{
    GstElement *pipeline;
    GstBus *bus;
    GstMessage *msg;

    /* Initialize GStreamer */
    gst_init (&argc, &argv);

    char pipeline_target[4096];
    printf("%s\n", PIPELINE_STR_DESCRIPTION);
    snprintf(pipeline_target, PIPELINE_STR_DESCRIPTION, 
    "NO20230128-115104-009260F.MP4", "fullres_sink", 640, 480, "smallres_sink");

    /* Build the pipeline */
    pipeline =
        gst_parse_launch
        ("playbin uri=NO20230128-115104-009260F.MP4",
        NULL);

    /* Start playing */
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait until error or EOS */
    bus = gst_element_get_bus (pipeline);
    msg =
        gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE,
        GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

    /* See next tutorial for proper error message handling/parsing */
    if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR) {
        g_error ("An error occurred! Re-run with the GST_DEBUG=*:WARN environment "
            "variable set for more details.");
    }

    /* Free resources */
    gst_message_unref (msg);
    gst_object_unref (bus);
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (pipeline);
    return 0;
}

int
main (int argc, char *argv[])
{
    return tutorial_main (argc, argv);
}

