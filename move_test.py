from xarm_control import XarmController
from keybrd import is_pressed
import time

LIN_VEL = 300  # translation speed
ROT_VEL = 15   # rotation speed (much slower)

if __name__ == '__main__':
    arm = XarmController('192.168.1.200')
    try:
        while arm.alive:
            dx = -1 if is_pressed('d') else 1 if is_pressed('a') else 0
            dy = -1 if is_pressed('w') else 1 if is_pressed('s') else 0
            dz = 1 if is_pressed('z') else -1 if is_pressed('x') else 0
            rx = 1 if is_pressed('u') else -1 if is_pressed('j') else 0
            ry = 1 if is_pressed('i') else -1 if is_pressed('k') else 0
            dw = 1 if is_pressed('q') else -1 if is_pressed('e') else 0
            vels = [dx*LIN_VEL, dy*LIN_VEL, dz*LIN_VEL, rx*ROT_VEL, ry*ROT_VEL, dw*ROT_VEL]
            arm.set_cartesian_velocity(vels)
            time.sleep(0.05)
    finally:
        arm.close()