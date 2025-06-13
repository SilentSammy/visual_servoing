import os
import cv2
import time
import numpy as np
from video import show_frame

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

    try:
        oi = v.BlackObjectIsolator()
        ol = v.ObjectLocator(obj_isolator=oi)
        al = v.ArucoLocator()
        controller = v.ArmController(obj_loc=ol)

        arm = XarmController('192.168.1.207')
        lin_vel = 100  # translation speed
        ang_vel = 30   # rotation speed (much slower)

        re = kb.rising_edge
        pr = kb.is_pressed

        while True:
            # Get current frame(s)
            color_frame = get_color_frame()
            depth_frame = get_depth_frame()

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
                arm.set_cartesian_velocity([x_vel, z_vel, y_vel, 0, 0, 0])

            show_frame(drawing_frame, "Drawing Frame", scale=0.5)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arm.close()
