import glob
import cv2
import numpy as np

# grab sorted lists of filenames
color_files = sorted(glob.glob(r"recording\color_pngs/color_*.png"))
depth_files = sorted(glob.glob(r"recording\depth_pngs/depth_*.png"))

for c_fn, d_fn in zip(color_files, depth_files):
    # 1) load color (8-bit BGR)
    color = cv2.imread(c_fn, cv2.IMREAD_COLOR)

    # 2) load depth (16-bit grayscale)
    depth = cv2.imread(d_fn, cv2.IMREAD_UNCHANGED)  # dtype=uint16

    # 3) visualize depth as a Jet colormap
    depth_vis = cv2.applyColorMap(
        (depth.astype(np.float32) / np.max(depth) * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )

    # 4) display
    cv2.imshow("Color", color)
    cv2.imshow("Depth (colormap)", depth_vis)

    if cv2.waitKey(1000 // 15) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
