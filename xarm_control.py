from xarm.wrapper import XArmAPI
import time
import traceback

class XarmController:
    def __init__(self, ip='192.168.1.207'):
        self._arm = XArmAPI(ip, baud_checkset=False)
        self.alive = True
        self._robot_init()

    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(5)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)

    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            print(f'err={data["error_code"]}, quit')
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            print('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    def set_cartesian_velocity(self, vels):
        """
        vels: [dx, dy, dz, rx, ry, rz]
        """
        code = self._arm.vc_set_cartesian_velocity(vels)
        if code != 0:
            print(f"Failed to set velocity, code={code}")
        return code

    def close(self):
        self.alive = False
        self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.release_state_changed_callback(self._state_changed_callback)
        self._arm.disconnect()

# No main loop or keyboard logic here!