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

class MultiVideoPlayer:
    def __init__(self, frame_sources):
        if not isinstance(frame_sources, (list, tuple)):
            raise ValueError("frame_sources must be a list or tuple of sources")
        self.frame_sources = frame_sources
        self.players = [VideoPlayer(src) for src in frame_sources]
        self.frame_count = min(p.frame_count for p in self.players)
        self._frame_idx = 0.0
        self.fps = min(p.fps for p in self.players)
        self.last_time = None
        self.dt = 0.0
        self.first_time = True
        self.scale = 1.0

    def show_frame(self, img, name, scale=None):
        scale = scale or self.scale
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        if self.first_time:
            self.first_time = False
            cv2.resizeWindow(name, int(img.shape[1]*scale), int(img.shape[0]*scale))
        cv2.imshow(name, img)
        if cv2.waitKey(1) & 0xFF == 27:
            raise KeyboardInterrupt

    def get_frames(self, idx=None):
        if idx is None:
            idx = self.frame_idx
        return tuple(p.get_frame(idx) for p in self.players)

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

if __name__ == "__main__":
    import keybrd as kb
    import visual_servoing as v

    # ---- CONFIGURATION ----
    # Set these to any valid source: int (webcam), str (folder or file), or None for depth
    COLOR_SOURCE = 0  # Webcam (use 0), or r"recording\color_pngs", or "video.mp4"
    DEPTH_SOURCE = None  # r"recording\depth_pngs", "depth_video.mp4", or None

    color_vp = VideoPlayer(COLOR_SOURCE)
    depth_vp = VideoPlayer(DEPTH_SOURCE) if DEPTH_SOURCE else None

    oi = v.BlackObjectIsolator()
    ol = v.ObjectLocator(obj_isolator=oi)
    oi_pipeline = [
        ("raw_frame", lambda: None),
        ("value_thres", lambda: oi.hsv_thres(color_frame, drawing_frame=drawing_frame)),
        ("remove_at_edges", lambda: oi.remove_at_edges(color_frame, drawing_frame=drawing_frame)),
        ("get_mask", lambda: oi.get_mask(color_frame, drawing_frame=drawing_frame)),
    ]
    ol_pipeline = [
        ("position", lambda: ol.get_position(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
        ("roll", lambda: ol.get_roll(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
        ("pose", lambda: ol.get_pose(color_frame, depth_frame=depth_frame, drawing_frame=drawing_frame)),
    ]
    pipeline = oi_pipeline + ol_pipeline

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

        color_vp.show_frame(drawing_frame, "Drawing Frame", scale=0.5)
