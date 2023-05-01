import cv2
import sys
import time

decoding_hal = {
    "basic": {
        "decoder": "libde265dec",
        "resizer":"videoscale",
        "resize_format": "",
        "compositor": "compositor"
    },
    "nvidia": {
        "decoder":"nvv4l2decoder",
        "resizer":"nvvidconv",
        "resize_format": ",BGRx",
        "compositor": "compositor"
    }
}

def main(source_path, hal_name, testing=False):
    gstreamer_line = \
        f"filesrc location={source_path} ! " \
        f"qtdemux name=demux demux.video_0 ! queue ! " \
        f"h265parse ! " \
        f"{decoding_hal[hal_name]['decoder']} ! " \
        f"tee name=decoded ! queue ! " \
        f"nvvidconv ! video/x-raw ! " \
        f"{decoding_hal[hal_name]['compositor']} name=comp sink_1::ypos=1944 ! " \
        f"appsink " \
        f"decoded. ! queue ! " \
        f"{decoding_hal[hal_name]['resizer']} ! video/x-raw,width=640,height=480  ! " \
        f"comp. "
    
    print(gstreamer_line)
    cap = cv2.VideoCapture(gstreamer_line, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise SystemExit('ERROR: failed to open the input video file!')
    
    next_timestamp = 0
    processing_timestamp = 0

    frame_ctr = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if frame is None:
            break

        if processing_timestamp - next_timestamp >= 1 / 30:
            print("Drop frame")
            """Detect objects in the input image."""

        finish_time = time.time()
        print("Timings ", 1 / 30, finish_time - start_time)
        next_timestamp += 1 / 30
        processing_timestamp = finish_time - start_time
        if 1 / 30 > (finish_time - start_time):
            time.sleep(1 / 30 - (finish_time - start_time))
        if testing:
            cv2.imwrite(f"full_res/img{frame_ctr:05}.jpg", frame)
        frame_ctr += 1
    print("Total frames ", frame_ctr)


if __name__ == '__main__':
    main(
        "NO20230128-115104-009260F.MP4",
        "nvidia",
        True
    )
