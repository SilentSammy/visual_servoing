import math
import cv2
import numpy as np
from simple_pid import PID

class BlackObjectIsolator:
    """ Basically creates a mask for the black object in the color frame. """
    def __init__(self,
        hue_range=(0, 180),                # Range of hue values to consider as "black"
        saturation_range=(0, 100),         # Range of saturation values to consider as "black"
        value_range=(0, 60),               # Range of value (brightness) to consider as "black"
        
        blur_kernel_size=(7, 7),                # Kernel size for GaussianBlur
        adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding method
        threshold_type=cv2.THRESH_BINARY_INV,   # Thresholding type
        block_size=141,                         # Size of the neighborhood used for thresholding (must be odd)
        c_value=6,                              # Constant subtracted from the mean or weighted mean (the higher the value, the darker the pixels need to be to be considered black)
        
        morph_kernel_size=(5, 5),   # Kernel size for morphological operations
        erosion_iterations=1,       # Number of iterations for erosion
        dilation_iterations=1,      # Number of iterations for dilation
        min_area=1000,              # Minimum area of contours to consider

        depth_range = (300, 1000),  # Range of depth values to consider as valid (in mm, for example)
    ):
        # HSV thresholding parameters
        self.hue_range = hue_range
        self.saturation_range = saturation_range
        self.value_range = value_range
        
        # Adaptive thresholding parameters
        self.blur_kernel_size = blur_kernel_size
        self.adaptive_method = adaptive_method
        self.threshold_type = threshold_type
        self.block_size = block_size
        self.c_value = c_value

        # Noise reduction parameters
        self.morph_kernel_size = morph_kernel_size
        self.erosion_iterations = erosion_iterations
        self.dilation_iterations = dilation_iterations
        self.min_area = min_area

        # Depth filtering parameters
        self.depth_range = depth_range  # Range of depth values to consider as valid (in mm, for example)

    def adaptive_thres(self, frame, drawing_frame=None):
        mask = adaptive_thres(frame, drawing_frame=drawing_frame,
                              blur_kernel_size=self.blur_kernel_size,
                              adaptive_method=self.adaptive_method,
                              threshold_type=self.threshold_type,
                              block_size=self.block_size,
                              c_value=self.c_value)
        return mask

    def hsv_thres(self, frame, drawing_frame=None):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, 
            (self.hue_range[0], self.saturation_range[0], self.value_range[0]),
            (self.hue_range[1], self.saturation_range[1], self.value_range[1])
        )
        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return mask

    def remove_at_edges(self, frame, drawing_frame=None):
        mask = self.hsv_thres(frame)

        # Get contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out contours that are touching the edges
        height, width = mask.shape
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x > 0 and y > 0 and (x + w) < width and (y + h) < height:
                filtered_contours.append(contour)
        # Create a mask for the filtered contours
        filtered_mask = np.zeros_like(mask)
        cv2.drawContours(filtered_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
        return filtered_mask

    def reduce_noise(self, frame, drawing_frame=None):
        mask = self.remove_at_edges(frame)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
        mask = cv2.erode(mask, kernel, iterations=self.erosion_iterations)
        mask = cv2.dilate(mask, kernel, iterations=self.dilation_iterations)

        # Filter out small contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask)
        for contour in contours:
            if cv2.contourArea(contour) >= self.min_area:
                cv2.drawContours(filtered_mask, [contour], -1, 255, thickness=cv2.FILLED)
        mask = filtered_mask

        if drawing_frame is not None:
            drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return mask

    def get_single_contour(self, frame=None, mask=None, drawing_frame=None):
        # Get the mask for the black object in the color frame
        mask = mask if mask is not None else self.reduce_noise(frame)

        # Define sorting function for contours
        def sort_contours(cnts):
            # Simple sorting by area, could be extended to sort by other criteria
            return sorted(cnts, key=lambda c: cv2.contourArea(c), reverse=True)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Sort contours and take the largest one
            sorted_contours = sort_contours(contours)
            largest_contour = sorted_contours[0]
            # Create a mask for the largest contour
            single_contour_mask = np.zeros_like(mask)
            cv2.drawContours(single_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            if drawing_frame is not None:
                bgr_mask = cv2.cvtColor(single_contour_mask, cv2.COLOR_GRAY2BGR)
                if bgr_mask.shape[:2] != drawing_frame.shape[:2]:
                    bgr_mask = cv2.resize(bgr_mask, (drawing_frame.shape[1], drawing_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                drawing_frame[:] = bgr_mask
            return single_contour_mask
        else:
            if drawing_frame is not None:
                blank = np.zeros_like(mask)
                bgr_blank = cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)
                if bgr_blank.shape[:2] != drawing_frame.shape[:2]:
                    bgr_blank = cv2.resize(bgr_blank, (drawing_frame.shape[1], drawing_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                drawing_frame[:] = bgr_blank
            return np.zeros_like(mask)

    def refine_with_depth(self, color_frame, depth_frame, drawing_frame):
        mask = self.reduce_noise(color_frame, drawing_frame=None)  # Don't draw yet

        # Resize mask to match depth_frame if needed
        color_h, color_w = color_frame.shape[:2]
        depth_h, depth_w = depth_frame.shape[:2]
        if (color_h, color_w) != (depth_h, depth_w):
            mask_resized = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        # Only keep mask pixels where depth is in range or depth==0
        min_depth, max_depth = self.depth_range
        valid = ((depth_frame >= min_depth) & (depth_frame <= max_depth)) | (depth_frame == 0)
        filtered_mask = np.where((mask_resized > 0) & valid, 255, 0).astype(np.uint8)

        # Now resize filtered_mask back to color frame size for downstream processing
        if (color_h, color_w) != (depth_h, depth_w):
            filtered_mask_color = cv2.resize(filtered_mask, (color_w, color_h), interpolation=cv2.INTER_NEAREST)
        else:
            filtered_mask_color = filtered_mask

        # Optionally draw
        if drawing_frame is not None:
            bgr_mask = cv2.cvtColor(filtered_mask_color, cv2.COLOR_GRAY2BGR)
            if bgr_mask.shape[:2] != drawing_frame.shape[:2]:
                bgr_mask = cv2.resize(bgr_mask, (drawing_frame.shape[1], drawing_frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            drawing_frame[:] = bgr_mask

        return filtered_mask_color

class ObjectLocator:
    """ Gets pose and other pose information about the object. """
    # Text formatting options for drawing
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_COLOR = (255, 255, 255)
    FONT_THICKNESS = 2
    X_POS = 10
    Y_START = 30
    Y_STEP = 30

    def __init__(self,
        obj_isolator=None,
        major_axis_ratio=2.2  # Ratio of the major axis to the minor axis for the object
    ):
        self.obj_isolator = obj_isolator or BlackObjectIsolator()
        self.major_axis_ratio = 2.2
    
    def get_object_mask(self, color_frame, depth_frame=None, drawing_frame=None):
        """ Returns a mask for the black object in the color frame. If depth_frame is provided, it refines the mask with depth information. """
        # Optionally refine with depth information
        if depth_frame is not None:
            mask = self.obj_isolator.refine_with_depth(color_frame, depth_frame, drawing_frame=drawing_frame)
        else:
            mask = self.obj_isolator.get_single_contour(color_frame, drawing_frame=drawing_frame)

        return mask

    def get_position(self, color_frame = None, mask = None, depth_frame=None, drawing_frame=None):
        """ Uses the mask from the ObjectIsolator to get x, y position of the object in the color frame;
        if depth_frame is provided, it also gets the z position of the object in the depth frame. """

        # Get the mask for the black object in the color frame. We can safely assume this is a single contour.
        mask =  mask if mask is not None else self.get_object_mask(color_frame=color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)
        moments = cv2.moments(mask, binaryImage=True)
        if moments['m00'] == 0:
            if drawing_frame is not None:
                # Optionally clear or annotate the drawing_frame here if desired
                pass
            return None

        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])

        height, width = mask.shape
        cx_norm = cx / width
        cy_norm = cy / height

        z = None
        if depth_frame is not None:
            # Map mask to depth frame size if needed
            color_h, color_w = color_frame.shape[:2]
            depth_h, depth_w = depth_frame.shape[:2]
            if (color_h, color_w) != (depth_h, depth_w):
                mask_resized = cv2.resize(mask, (depth_w, depth_h), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask

            # Select depth pixels inside the mask and not zero (any zero depth pixels are probably just in the dead zone of the depth camera)
            mask_bool = (mask_resized > 0)
            depth_pixels = depth_frame[mask_bool]
            valid_depths = depth_pixels[depth_pixels > 0]
            if valid_depths.size > 0:
                z = float(np.mean(valid_depths))

        # Display
        if drawing_frame is not None:
            # Draw a crosshair at the centroid
            cv2.drawMarker(drawing_frame, (cx, cy), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

            # Write x y z in the top left corner
            cv2.putText(drawing_frame, f"x: {cx_norm:.2f}", (self.X_POS, self.Y_START), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
            cv2.putText(drawing_frame, f"y: {cy_norm:.2f}", (self.X_POS, self.Y_START + self.Y_STEP), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
            if z is not None:
                cv2.putText(drawing_frame, f"z: {float(z):.2f}", (self.X_POS, self.Y_START + 2 * self.Y_STEP), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)

        return cx_norm, cy_norm, z

    def get_roll(self, color_frame=None, mask=None, depth_frame=None, drawing_frame=None):
        roll = None

        # We can safely assume that the mask is a single contour.
        mask =  mask if mask is not None else self.get_object_mask(color_frame=color_frame, depth_frame=depth_frame)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        contour = contours[0]

        # Get contour line info
        line = get_contour_major_axis(contour)
        pt1, pt2, center, angle, length = line
        roll = angle  # Assuming roll is the angle of the contour line
        if drawing_frame is not None:
            # Draw the contour line
            cv2.line(drawing_frame, pt1, pt2, (0, 255, 0), 2)
            # Draw the center point
            cv2.circle(drawing_frame, center, 5, (0, 0, 255), -1)

            # Write the angle in the top left corner
            if roll is not None:
                cv2.putText(drawing_frame, f"roll: {math.degrees(roll):.2f}", (self.X_POS, self.Y_START + 3 * self.Y_STEP), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
        
        return roll, line

    def get_orientation(self, color_frame=None, mask=None, depth_frame=None, drawing_frame=None):
        mask = mask if mask is not None else self.get_object_mask(color_frame=color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)
        # Get the roll of the object
        roll, line = self.get_roll(color_frame=color_frame, mask=mask, depth_frame=depth_frame, drawing_frame=drawing_frame)
        if roll is None:
            return None, None, None

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        
        contour = contours[0]
        major_ax = get_contour_major_axis(contour)
        minor_ax = get_contour_minor_axis(contour)
        L_major = major_ax[-1]
        L_minor = minor_ax[-1]
        R0 = self.major_axis_ratio
        # Avoid division by zero
        if L_minor == 0 or L_major == 0:
            pitch = None
            yaw = None
        else:
            # Clamp arguments to [-1, 1] for safety
            arg_pitch = L_minor / (L_major / R0)
            arg_yaw = L_major / (L_minor * R0)
            arg_pitch = max(-1.0, min(1.0, arg_pitch))
            arg_yaw = max(-1.0, min(1.0, arg_yaw))
            pitch = math.acos(arg_pitch)
            yaw = math.acos(arg_yaw)
        if drawing_frame is not None:
            draw_line(drawing_frame, major_ax)
            draw_line(drawing_frame, minor_ax)
            y_offset = self.Y_START + 3 * self.Y_STEP
            if roll is not None:
                cv2.putText(drawing_frame, f"roll: {math.degrees(roll):.2f}", (self.X_POS, y_offset), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
                y_offset += self.Y_STEP
            if pitch is not None:
                cv2.putText(drawing_frame, f"pitch: {math.degrees(pitch):.2f}", (self.X_POS, y_offset), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
                y_offset += self.Y_STEP
            if yaw is not None:
                cv2.putText(drawing_frame, f"yaw: {math.degrees(yaw):.2f}", (self.X_POS, y_offset), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
        return roll, pitch, yaw

    def get_pose(self, color_frame, depth_frame=None, drawing_frame=None):
        mask = self.obj_isolator.get_single_contour(color_frame)
        
        # Get the pose info of the object
        pos = self.get_position(color_frame, mask=mask, depth_frame=depth_frame)
        if pos is None:
            return None
        cx_norm, cy_norm, z = pos
        roll, line = self.get_roll(color_frame, mask=mask, depth_frame=depth_frame, drawing_frame=drawing_frame)
        # ...rest of your code...

        if cx_norm is None or cy_norm is None:
            return None

        if drawing_frame is not None:
            draw_line(drawing_frame, line)

            # Write the pose information in the top left corner
            cv2.putText(drawing_frame, f"x: {cx_norm:.2f}", (self.X_POS, self.Y_START), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
            cv2.putText(drawing_frame, f"y: {cy_norm:.2f}", (self.X_POS, self.Y_START + self.Y_STEP), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
            if z is not None:
                cv2.putText(drawing_frame, f"z: {float(z):.2f}", (self.X_POS, self.Y_START + 2 * self.Y_STEP), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
            cv2.putText(drawing_frame, f"roll: {math.degrees(roll):.2f}", (self.X_POS, self.Y_START + 3 * self.Y_STEP), self.FONT, self.FONT_SCALE, self.FONT_COLOR, self.FONT_THICKNESS)
        return cx_norm, cy_norm, z, roll

class ArucoLocator:
    """Locates a single ArUco marker in a color+depth frame, returning (x, y, z, roll)."""

    # Text formatting for drawing
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7
    FONT_COLOR = (0, 255, 0)
    FONT_THICKNESS = 2
    X_POS = 10
    Y_START = 30
    Y_STEP = 30

    def __init__(self,
                 dictionary=None,
                 parameters=None):
        # use 4x4_50 if no dictionary passed
        self.aruco_dict = (dictionary or cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        params = parameters or cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, params)

    def get_position(self, color_frame, depth_frame=None, drawing_frame=None):
        # detect markers
        corners, ids, _ = self.detector.detectMarkers(color_frame)
        if ids is None or len(ids) == 0:
            return None, None, None

        # take the first detected marker
        pts = np.asarray(corners[0], dtype=np.float32)
        # handle shapes (1,4,2) or (4,1,2)
        if pts.ndim == 3:
            if pts.shape[0] == 1:
                pts = pts[0]
            elif pts.shape[1] == 1:
                pts = pts[:, 0, :]
        pts = pts.reshape(-1, 2)

        # centroid in pixel coords
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        h, w = color_frame.shape[:2]
        cx_norm = cx / w
        cy_norm = cy / h

        # estimate depth if provided
        z = None
        if depth_frame is not None:
            # assume depth_frame same size as color_frame
            mask = np.zeros_like(depth_frame, dtype=np.uint8)
            poly = pts.astype(np.int32)
            cv2.fillConvexPoly(mask, poly, 255)
            depths = depth_frame[mask > 0]
            valid = depths > 0
            if valid.any():
                z = float(depths[valid].mean())

        # drawing
        if drawing_frame is not None:
            cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
            center = (int(round(cx)), int(round(cy)))
            cv2.drawMarker(drawing_frame, center,
                           (0, 255, 0),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=20, thickness=2)
            cv2.putText(drawing_frame,
                        f"x: {cx_norm:.2f}",
                        (self.X_POS, self.Y_START),
                        self.FONT, self.FONT_SCALE,
                        self.FONT_COLOR, self.FONT_THICKNESS)
            cv2.putText(drawing_frame,
                        f"y: {cy_norm:.2f}",
                        (self.X_POS, self.Y_START + self.Y_STEP),
                        self.FONT, self.FONT_SCALE,
                        self.FONT_COLOR, self.FONT_THICKNESS)
            if z is not None:
                cv2.putText(drawing_frame,
                            f"z: {z:.2f}",
                            (self.X_POS, self.Y_START + 2 * self.Y_STEP),
                            self.FONT, self.FONT_SCALE,
                            self.FONT_COLOR, self.FONT_THICKNESS)

        return cx_norm, cy_norm, z

    def get_roll(self, color_frame, drawing_frame=None):
        # detect markers again
        corners, ids, _ = self.detector.detectMarkers(color_frame)
        if ids is None or len(ids) == 0:
            return None, None

        # take first marker corners
        pts = np.asarray(corners[0], dtype=np.float32)
        if pts.ndim == 3:
            if pts.shape[0] == 1:
                pts = pts[0]
            elif pts.shape[1] == 1:
                pts = pts[:, 0, :]
        pts = pts.reshape(-1, 2)

        # roll = angle of edge from corner0→corner1
        p0 = pts[0]
        p1 = pts[1]
        angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

        # drawing
        pt1 = (int(round(p0[0])), int(round(p0[1])))
        pt2 = (int(round(p1[0])), int(round(p1[1])))
        if drawing_frame is not None:
            cv2.line(drawing_frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(drawing_frame,
                        f"roll: {math.degrees(angle):.1f}°",
                        (self.X_POS, self.Y_START + 3 * self.Y_STEP),
                        self.FONT, self.FONT_SCALE,
                        self.FONT_COLOR, self.FONT_THICKNESS)

        return angle, (pt1, pt2)

    def get_pose(self, color_frame, depth_frame=None, drawing_frame=None):
        """
        Returns (x_norm, y_norm, z, roll) just like ObjectLocator.get_pose.
        """
        cx, cy, z = self.get_position(color_frame,
                                      depth_frame=depth_frame,
                                      drawing_frame=drawing_frame)
        roll, line = self.get_roll(color_frame,
                                   drawing_frame=drawing_frame)
        if cx is None or cy is None:
            return None, None, None, None

        # Optionally annotate the line (as ObjectLocator does)
        if drawing_frame is not None and line is not None:
            cv2.line(drawing_frame, line[0], line[1], (0, 255, 0), 2)

        return cx, cy, z, roll

class ArmController:
    def __init__(self,
            obj_loc=None,
            aruco_loc=None,
            target_offset=(0.0, -0.3, 0.0),
            target_local=True
        ):
        """
        Args:
            obj_loc: ObjectLocator instance or None.
            target_offset: Tuple (dx, dy, dz) offset from detected object position.
            target_local: If True, offset is applied in the object's local frame (rotated by roll).
        """
        self.obj_loc = obj_loc or ObjectLocator()
        self.aruco_loc = aruco_loc or ArucoLocator()
        self.target_offset = target_offset
        self.target_local = target_local
        self.x_pid = PID(10.0, 0.1, 0.05, setpoint=0, output_limits=(-1, 1))
        self.y_pid = PID(10.0, 0.1, 0.05, setpoint=0, output_limits=(-1, 1))
        self.z_pid = PID(10.0, 0.1, 0.05, setpoint=0, output_limits=(-1, 1))
        self.roll_pid = PID(10.0, 0.1, 0.05, setpoint=0, output_limits=(-1, 1))

    def get_target(self, color_frame, depth_frame=None, drawing_frame=None):
        object_pose = self.obj_loc.get_pose(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)
        if object_pose is None:
            return None
        
        cx_norm, cy_norm, z, roll = object_pose

        # Apply the target offset, rotating if target_local is True
        dx, dy, dz = self.target_offset
        if self.target_local and roll is not None:
            # Rotate (dx, dy) by roll
            dx_rot = dx * math.cos(roll) - dy * math.sin(roll)
            dy_rot = dx * math.sin(roll) + dy * math.cos(roll)
        else:
            dx_rot, dy_rot = dx, dy

        target_x = cx_norm + dx_rot
        target_y = cy_norm + dy_rot
        target_z = z + dz if z is not None else None

        if drawing_frame is not None:
            h, w = color_frame.shape[:2]
            center = (int(round(target_x * w)), int(round(target_y * h)))
            size = 20
            color = (255, 0, 0)
            thickness = 2

            if self.target_local and roll is not None:
                # Draw rotated crosshair
                for angle_offset in [0, math.pi / 2]:
                    angle = roll + angle_offset
                    dx = int(size * math.cos(angle) / 2)
                    dy = int(size * math.sin(angle) / 2)
                    pt1 = (center[0] - dx, center[1] - dy)
                    pt2 = (center[0] + dx, center[1] + dy)
                    cv2.line(drawing_frame, pt1, pt2, color, thickness)
            else:
                # Draw standard crosshair
                cv2.drawMarker(drawing_frame, center, color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)

            # Write the target position in the top left corner
            target_z_str = f"{target_z:.2f}" if target_z is not None else "None"
            cv2.putText(drawing_frame, f"Target: ({target_x:.2f}, {target_y:.2f}, {target_z_str})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return target_x, target_y, target_z, roll if self.target_local else 0

    def compute_error(self, color_frame, depth_frame=None, drawing_frame=None):
        target_pose = self.get_target(color_frame, depth_frame=depth_frame)
        current_pose = self.aruco_loc.get_pose(color_frame, depth_frame=depth_frame)

        if drawing_frame is not None:
            # Draw the target position
            if target_pose is not None:
                cx, cy, cz, roll = target_pose
                if cx is not None and cy is not None:
                    h, w = color_frame.shape[:2]
                    center = (int(round(cx * w)), int(round(cy * h)))
                    cv2.drawMarker(drawing_frame, center, (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cz_str = f"{cz:.2f}" if cz is not None else "None"
                cv2.putText(drawing_frame, f"Target: ({cx if cx is not None else 'None'},"
                                        f" {cy if cy is not None else 'None'}, {cz_str})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if current_pose is not None:
                cx, cy, cz, roll = current_pose
                if cx is not None and cy is not None:
                    h, w = color_frame.shape[:2]
                    center = (int(round(cx * w)), int(round(cy * h)))
                    cv2.drawMarker(drawing_frame, center, (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
                cz_str = f"{cz:.2f}" if cz is not None else "None"
                cv2.putText(drawing_frame, f"Current: ({cx if cx is not None else 'None'}," 
                                        f" {cy if cy is not None else 'None'}, {cz_str})",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if target_pose is None or current_pose is None:
            return None
        
        target_x, target_y, target_z, target_roll = target_pose
        current_x, current_y, current_z, current_roll = current_pose
        # Compute the error as the difference between target and current positions
        error_x = target_x - current_x if target_x is not None and current_x is not None else None
        error_y = target_y - current_y if target_y is not None and current_y is not None else None
        error_z = target_z - current_z if target_z is not None and current_z is not None else None
        error_roll = target_roll - current_roll if target_roll is not None and current_roll is not None else None

        if drawing_frame is not None:
            # Draw line between target and current positions
            if target_x is not None and target_y is not None and current_x is not None and current_y is not None:
                h, w = color_frame.shape[:2]
                target_center = (int(round(target_x * w)), int(round(target_y * h)))
                current_center = (int(round(current_x * w)), int(round(current_y * h)))
                cv2.line(drawing_frame, target_center, current_center, (255, 255, 0), 2)
                error_x_str = f"{error_x:.2f}" if error_x is not None else "None"
                error_y_str = f"{error_y:.2f}" if error_y is not None else "None"
                error_z_str = f"{error_z:.2f}" if error_z is not None else "None"
                error_roll_str = f"{error_roll:.2f}" if error_roll is not None else "None"
                cv2.putText(
                    drawing_frame,
                    f"Error: ({error_x_str}, {error_y_str}, {error_z_str}, {error_roll_str})",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
                )

        return error_x, error_y, error_z, error_roll
    
    def cartesian_vels(self, color_frame, depth_frame=None, drawing_frame=None):
        errors = self.compute_error(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)
        if errors is None:
            return 0, 0, 0, 0
        error_x, error_y, error_z, error_roll = errors
        error_z = error_z or 0

        # Compute the control signals using PID controllers
        control_x = self.x_pid(error_x) if error_x is not None else 0
        control_y = self.y_pid(error_y) if error_y is not None else 0
        control_z = self.z_pid(error_z) if error_z is not None else 0
        control_roll = self.roll_pid(error_roll) if error_roll is not None else 0

        return control_x, control_y, control_z, control_roll
        
def adaptive_thres(frame, drawing_frame=None,
    blur_kernel_size=(7, 7),  # Kernel size for GaussianBlur
    adaptive_method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive thresholding method
    threshold_type=cv2.THRESH_BINARY_INV,  # Thresholding type
    block_size=141,  # Size of the neighborhood used for thresholding (must be odd)
    c_value=6,  # Constant subtracted from the mean or weighted mean (the higher the value, the darker the pixels need to be to be considered black)
):
    # Processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, blur_kernel_size, 0)
    mask = cv2.adaptiveThreshold(gray, 255, adaptive_method, threshold_type, block_size, c_value)
    if drawing_frame is not None:
        drawing_frame[:] = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask

def get_contour_major_axis(c):
    # Fit a line to the contour
    vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    
    # Project contour points onto the line's direction vector.
    projections = [((int(pt[0][0]) - int(cx)) * vx + (int(pt[0][1]) - int(cy)) * vy) for pt in c]
    min_proj = min(projections)
    max_proj = max(projections)
    
    # Compute endpoints from the extreme projection values.
    pt1 = (int(round(cx + vx * min_proj)), int(round(cy + vy * min_proj)))
    pt2 = (int(round(cx + vx * max_proj)), int(round(cy + vy * max_proj)))
    
    # Calculate the line angle in radians.
    angle = math.atan2(vy, vx)
    
    # Calculate the line length given pt1 and pt2.
    length = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    
    # Ensure cx, cy are ints as well
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    center = (cx_int, cy_int)
    
    return pt1, pt2, center, angle, length

def draw_line(drawing_frame, line):
    # Unpack the line information
    pt1, pt2, center, angle, length = line

    # Draw the line on the drawing frame
    cv2.line(drawing_frame, pt1, pt2, (0, 255, 0), 2)
    # Draw the center point
    cv2.circle(drawing_frame, center, 5, (0, 0, 255), -1)

def get_contour_minor_axis(c):
    # Fit a line to the contour (major axis)
    vx, vy, cx, cy = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    # The minor axis is perpendicular to the major axis
    # So, rotate (vx, vy) by 90 degrees
    perp_vx = -vy
    perp_vy = vx

    # Project contour points onto the perpendicular direction
    projections = [((int(pt[0][0]) - int(cx)) * perp_vx + (int(pt[0][1]) - int(cy)) * perp_vy) for pt in c]
    min_proj = min(projections)
    max_proj = max(projections)

    # Compute endpoints from the extreme projection values
    pt1 = (int(round(cx + perp_vx * min_proj)), int(round(cy + perp_vy * min_proj)))
    pt2 = (int(round(cx + perp_vx * max_proj)), int(round(cy + perp_vy * max_proj)))

    # Calculate the line angle in radians
    angle = math.atan2(perp_vy, perp_vx)

    # Calculate the line length given pt1 and pt2
    length = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

    # Ensure cx, cy are ints as well
    cx_int = int(round(cx))
    cy_int = int(round(cy))
    center = (cx_int, cy_int)

    return pt1, pt2, center, angle, length