import os
import cv2
import time
import numpy as np
from video import show_frame, VideoRecorder

cap = cv2.VideoCapture(0)
def get_color_frame():
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def get_depth_frame():
    return None # Placeholder for depth frame, replace with actual depth capture logic
        
if __name__ == "__main__":
    import keybrd as kb
    import visual_servoing as v
    from xarm_control import XarmController
    from kinect import get_frames

    try:
        vr = VideoRecorder(dir_path="./recording/videos", fps=30)
        oi = v.BlackObjectIsolator()
        ol = v.ObjectLocator(obj_isolator=oi)
        al = v.ArucoLocator()
        controller = v.ArmController(obj_loc=ol, target_offset=(0, -0.175, 0))  # Adjust target offset as needed

        arm = XarmController('192.168.1.200')
        lin_vel = 200  # translation speed
        ang_vel = 15   # rotation speed (much slower)

        re = kb.rising_edge
        pr = kb.is_pressed

        while True:
            # Get current frame(s)
            # color_frame = get_color_frame()
            # depth_frame = get_depth_frame()
            color_frame, depth_frame = get_frames()
            # depth_frame = None

            if color_frame is None:
                continue
            
            drawing_frame = color_frame.copy()
            
            # Process control
            cart_vels = controller.cartesian_vels(color_frame=color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)
            if cart_vels is not None:
                x_vel = cart_vels[0] * lin_vel
                y_vel = cart_vels[1] * lin_vel
                z_vel = cart_vels[2] * lin_vel
                rx_vel = cart_vels[3] * ang_vel
                arm.set_cartesian_velocity([x_vel, z_vel, y_vel, 0, rx_vel, 0])

            # Optionally record the already annotated frame
            if kb.is_toggled('i'):
                if not vr.is_recording():
                    vr.start(drawing_frame)
                vr.write(drawing_frame)
            else:
                if vr.is_recording():
                    vr.stop()

            show_frame(drawing_frame, "Drawing Frame", scale=0.75)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.close()
