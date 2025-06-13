# kinect_module.py

import cv2
import numpy as np
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, ImageFormat, FPS

# Initialize the device with your Viewer parameters
_k4a = PyK4A(
    Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        color_format=ImageFormat.COLOR_MJPG,   # MJPG → needs decoding
        camera_fps=FPS.FPS_15,                # 15 FPS as in Viewer
        synchronized_images_only=True
    )
)
_k4a.start()

def get_frames():
    """
    Returns:
      - color_bgr: H×W×3 uint8 BGR image (decoded from MJPG)
      - depth_raw: H×W uint16 depth in millimeters
    """
    capture = _k4a.get_capture()

    # ——— Color ———
    raw = capture.color  # 1D uint8 JPEG buffer
    if raw is None:
        color_bgr = None
    else:
        # turn into a numpy array and decode JPEG
        jpg = np.frombuffer(raw, dtype=np.uint8)
        color_bgr = cv2.imdecode(jpg, cv2.IMREAD_COLOR)

    # ——— Depth ———
    depth_raw = capture.depth  # uint16 array

    return color_bgr, depth_raw

if __name__ == "__main__":
    try:
        while True:
            color, depth = get_frames()

            # Display Color
            if color is not None:
                cv2.imshow("Color", color)
            else:
                cv2.imshow("Color", np.zeros((480, 640, 3), dtype=np.uint8))

            # Display Depth (Jet colormap)
            if depth is not None:
                d8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imshow("Depth", cv2.applyColorMap(d8, cv2.COLORMAP_JET))
            else:
                cv2.imshow("Depth", np.zeros((576, 640, 3), dtype=np.uint8))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        _k4a.stop()
        cv2.destroyAllWindows()
