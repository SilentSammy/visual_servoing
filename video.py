import os
import cv2
import time
import numpy as np

class VideoPlayer:
    def __init__(self, frame_source):
        self.frame_source = frame_source
        self.frame_count = 0
        self._frame_idx = 0.0
        self.fps = 30  # Default FPS
        self._get_frame = None
        self.last_time = None
        self.dt = 0.0
        self.setup_video_source()
        self.first_time = True

    def show_frame(self, img, name, scale=1):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        if self.first_time:
            self.first_time = False
            cv2.resizeWindow(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
            self.first_time = False
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    def get_frame(self, idx=None):
        if idx is None:
            idx = self.frame_idx
        return self._get_frame(idx) # type: ignore

    def step(self, step_size=1):
        self._frame_idx += step_size
        self._frame_idx = self._frame_idx % self.frame_count
    
    def time_step(self):
        self.dt = time.time() - self.last_time if self.last_time is not None else 0.0
        self.last_time = time.time()
        return self.dt

    def move(self, speed=1):
        self._frame_idx += speed * self.dt * self.fps
        self._frame_idx = self._frame_idx % self.frame_count

    @property
    def frame_idx(self):
        return int(self._frame_idx)

    def setup_video_source(self):
        # If frame_source is a cv2.VideoCapture object, use it directly
        if isinstance(self.frame_source, cv2.VideoCapture):
            cap = self.frame_source
            if not cap.isOpened():
                print("Error opening video file")
                exit(1)
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames:", self.frame_count)
            
            def get_frame(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame", idx)
                    return None
                return frame
            
            self._get_frame = get_frame
        # If frame_source is a folder, load images
        elif os.path.isdir(self.frame_source):
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = sorted([
                os.path.join(self.frame_source, f) 
                for f in os.listdir(self.frame_source) 
                if f.lower().endswith(image_extensions)
            ])
            self.frame_count = len(image_files)
            print("Total frames (images):", self.frame_count)
            
            def get_frame(idx):
                idx = int(idx)
                if idx < 0 or idx >= len(image_files):
                    print("Index out of bounds:", idx)
                    return None
                # Use IMREAD_UNCHANGED for depth, IMREAD_COLOR for color
                # You may want to add a flag to VideoPlayer to distinguish color/depth
                frame = cv2.imread(image_files[idx], cv2.IMREAD_UNCHANGED)
                if frame is None:
                    print("Failed to load image", image_files[idx])
                return frame
            
            self._get_frame = get_frame
        else:
            # Assume frame_source is a video file.
            cap = cv2.VideoCapture(self.frame_source)
            if not cap.isOpened():
                print("Error opening video file:", self.frame_source)
                exit(1)
            
            self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("Total frames:", self.frame_count)
            
            def get_frame(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame", idx)
                    return None
                return frame
            
            self._get_frame = get_frame

class VideoRecorder:
    def __init__(self, dir_path="./resources/videos", fps=30):
        self.dir_path = dir_path
        self.fps = fps
        self.vw = None
        self.filename = None

    def start(self, frame):
        if self.vw is not None:
            self.stop()
        os.makedirs(self.dir_path, exist_ok=True)
        height, width = frame.shape[:2]
        self.filename = os.path.join(self.dir_path, time.strftime("output_%Y-%m-%d_%H-%M-%S.mp4"))
        fourcc = cv2.VideoWriter.fourcc(*'avc1')  # Use 'mp4v' for better compatibility
        self.vw = cv2.VideoWriter(self.filename, fourcc, self.fps, (width, height))
        if not self.vw.isOpened():
            print(f"Failed to open video writer for {self.filename}")
        else:
            print(f"Video recording started: {self.filename}")

    def write(self, frame):
        if self.vw is not None and frame is not None:
            self.vw.write(frame)

    def stop(self):
        if self.vw is not None:
            self.vw.release()
            print(f"Video recording closed: {self.filename}")
            self.vw = None
            self.filename = None

    def is_recording(self):
        return self.vw is not None

def show_frame(img, name, scale=1):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        cv2.resizeWindow(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

if __name__ == "__main__":
    import keybrd as kb
    import visual_servoing as v

    vr = VideoRecorder(dir_path="./recording/videos", fps=30)

    # ---- CONFIGURATION ----
    # Set these to any valid source: int (webcam), str (folder or file), or None for depth
    COLOR_SOURCE = r"recording\color_pngs"  # or "video.mp4", or None for webcam
    DEPTH_SOURCE = r"recording\depth_pngs"
    COLOR_SOURCE = 0
    DEPTH_SOURCE = None

    color_vp = VideoPlayer(COLOR_SOURCE)
    depth_vp = VideoPlayer(DEPTH_SOURCE) if DEPTH_SOURCE else None

    oi = v.BlackObjectIsolator()
    ol = v.ObjectLocator(obj_isolator=oi)
    al = v.ArucoLocator()
    controller = v.ArmController(obj_loc=ol)
    raw = [ ("raw_frame", lambda: None), ]
    oi_pipeline = [
        ("get_mask", lambda: oi.get_single_contour(color_frame, drawing_frame=drawing_frame)),
        ("refine_w_depth", lambda: oi.refine_with_depth(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
    ]
    ol_pipeline = [
        ("position", lambda: ol.get_position(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
        ("roll", lambda: ol.get_roll(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
        ("orientation", lambda: ol.get_orientation(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
        ("pose", lambda: ol.get_pose(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
    ]
    aruco_pipeline = [
        ("aruco_pos", lambda: al.get_position(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
        ("aruco_orientation", lambda: al.get_orientation(color_frame=color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
        ("aruco_pose", lambda: al.get_pose(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
    ]
    controller_pipeline = [
        ("get_target", lambda: controller.get_target(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
        ("compute_error", lambda: controller.compute_error(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
    ]
    pipeline = raw + aruco_pipeline + controller_pipeline

    re = kb.rising_edge
    pr = kb.is_pressed

    layer = 0
    while True:
        color_vp.time_step()
        color_vp.move(1 if pr('d') else -1 if pr('a') else 0)
        color_vp.move((1 if pr('e') else -1 if pr('q') else 0) * 10)
        color_vp.step(1 if re('w') else -1 if re('s') else 0)

        # Sync depth_vp to color_vp if present
        if depth_vp is not None:
            depth_vp._frame_idx = color_vp._frame_idx

        # Get current frame(s)
        color_frame = color_vp.get_frame()
        depth_frame = depth_vp.get_frame() if depth_vp is not None else None

        if color_frame is None:
            continue
        print(f"Frame {color_vp.frame_idx}/{color_vp.frame_count} ", end=': ')
        drawing_frame = color_frame.copy()

        for i in range(len(pipeline)):
            if re(str(i+1)):
                layer = i

        print(f"{pipeline[layer][0]}")
        pipeline[layer][1]()

        color_vp.show_frame(drawing_frame, "Drawing Frame", scale=0.75)
