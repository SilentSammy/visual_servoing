import os
import cv2
import numpy as np
from kinect import get_frames

# make folders for the two streams
os.makedirs("color_pngs", exist_ok=True)
os.makedirs("depth_pngs", exist_ok=True)

frame_idx = 0
try:
    while True:
        color, depth = get_frames()  # color: H×W×3 uint8, depth: H×W uint16

        if color is None or depth is None:
            print("No frame, stopping.")
            break

        # Save color as 8-bit PNG
        cv2.imwrite(f"color_pngs/color_{frame_idx:06d}.png", color)

        # Save depth as 16-bit PNG (mm preserved)
        cv2.imwrite(
            f"depth_pngs/depth_{frame_idx:06d}.png",
            depth,
            [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )

        print(f"Saved frame {frame_idx}")
        frame_idx += 1

        # Optional: show a live preview (press 'q' to stop early)
        cv2.imshow("Color Preview", color)
        depth_vis = cv2.applyColorMap(
            cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        cv2.imshow("Depth Preview", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    print("Done recording.")
