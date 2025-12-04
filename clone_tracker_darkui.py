"""
clone_tracker_full_tk.py
Full Clone Tracker with:
- Tkinter dark toolbar UI
- Neon clone (mirrored) skeleton
- Glow trails
- Background blur using MediaPipe SelfieSegmentation
- Simple gesture detection (left/right/both hands up, T-pose)
- Recording (VideoWriter)
- Optional 3D pose window (matplotlib) if available
- Designed for Python 3.10 / Windows; depends on opencv-python, mediapipe, numpy, matplotlib (optional)
"""

import os
import time
import math
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

# --- Try to import matplotlib for optional 3D plotting ---
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# --- MediaPipe setup ---
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_selfie = mp.solutions.selfie_segmentation

POSE = mp_pose.Pose(static_image_mode=False,
                    model_complexity=0,
                    smooth_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

SEG = mp_selfie.SelfieSegmentation(model_selection=1)

# skeletal connections subset (MediaPipe indices)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
    (25, 27), (26, 28), (27, 31), (28, 32)
]

# ----------------- Helpers -----------------
def get_gradient_color(t: float):
    """Return an RGB tuple (0-255) cycling through hues based on time t."""
    hue = (t * 120) % 360
    c = 1.0
    x = c * (1 - abs((hue / 60) % 2 - 1))
    if 0 <= hue < 60:
        r, g, b = c, x, 0
    elif 60 <= hue < 120:
        r, g, b = x, c, 0
    elif 120 <= hue < 180:
        r, g, b = 0, c, x
    elif 180 <= hue < 240:
        r, g, b = 0, x, c
    elif 240 <= hue < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (int(np.clip(r * 255, 0, 255)),
            int(np.clip(g * 255, 0, 255)),
            int(np.clip(b * 255, 0, 255)))

def draw_smooth_line(img, p1, p2, color, thickness=6):
    cv2.line(img, p1, p2, color, thickness, lineType=cv2.LINE_AA)
    outer = tuple(max(0, c // 3) for c in color)
    cv2.line(img, p1, p2, outer, max(1, thickness + 2), lineType=cv2.LINE_AA)

def draw_smooth_circle(img, center, radius, color):
    cv2.circle(img, center, radius, color, -1, lineType=cv2.LINE_AA)
    inner = max(1, radius // 2)
    cv2.circle(img, center, inner, (255,255,255), -1, lineType=cv2.LINE_AA)

def apply_background_blur(frame, mask, blur_strength=21):
    # mask: HxWx1 floats 0..1
    if mask is None:
        return frame
    mask_3c = np.dstack([mask, mask, mask])
    blurred = cv2.GaussianBlur(frame, (blur_strength|1, blur_strength|1), 0)
    composed = (frame * mask_3c + blurred * (1 - mask_3c)).astype(np.uint8)
    return composed

def simple_gestures(landmarks):
    """Return simple gestures dict using MediaPipe normalized landmarks list."""
    gest = {"left_hand_up": False, "right_hand_up": False, "both_hands_up": False, "t_pose": False}
    try:
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
    except Exception:
        return gest
    # y smaller = higher on image
    if left_wrist.y < left_shoulder.y - 0.05:
        gest["left_hand_up"] = True
    if right_wrist.y < right_shoulder.y - 0.05:
        gest["right_hand_up"] = True
    if gest["left_hand_up"] and gest["right_hand_up"]:
        gest["both_hands_up"] = True
    if (abs(left_wrist.y - left_shoulder.y) < 0.08 and
        abs(right_wrist.y - right_shoulder.y) < 0.08 and
        abs(left_wrist.x - left_shoulder.x) > 0.18 and
        abs(right_wrist.x - right_shoulder.x) > 0.18):
        gest["t_pose"] = True
    return gest

# ----------------- Tkinter Dark Theme -----------------
class DarkStyle:
    BG = "#111318"
    FG = "#E6EEF3"
    BTN = "#22272B"
    ACCENT = "#00B4FF"

# ----------------- Main App -----------------
class CloneTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clone Tracker - Dark UI")
        self.root.configure(bg=DarkStyle.BG)

        # state
        self.cap = None
        self.camera_index = 0
        self.width = 640
        self.height = 480
        self.running = False
        self.recording = False
        self.writer = None
        self.trail_length = 8
        self.trails = []  # list of lists of joint points per frame
        self.neon_intensity = 0.9
        self.use_blur = True
        self.use_3d = False and MATPLOTLIB_AVAILABLE
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.prev_time = time.time()
        self.fps_display = 0
        self.last_fps_update = time.time()

        # 3D figure if available
        if MATPLOTLIB_AVAILABLE:
            self.fig = plt.figure(figsize=(5,4))
            self.ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.fig = None
            self.ax = None

        # UI
        self.build_toolbar()
        self.build_video_area()
        self.build_statusbar()

        # schedule update
        self.update_loop()

    # ------------- UI builders -------------
    def build_toolbar(self):
        toolbar = tk.Frame(self.root, bg=DarkStyle.BG)
        toolbar.pack(fill="x", padx=6, pady=6)

        tk.Label(toolbar, text="Camera:", fg=DarkStyle.FG, bg=DarkStyle.BG).pack(side="left")
        self.cam_entry = tk.Entry(toolbar, width=4, bg=DarkStyle.BTN, fg=DarkStyle.FG, insertbackground=DarkStyle.FG)
        self.cam_entry.insert(0, "0")
        self.cam_entry.pack(side="left", padx=(4,8))

        tk.Button(toolbar, text="Open", bg=DarkStyle.BTN, fg=DarkStyle.FG, command=self.open_camera, relief="flat").pack(side="left", padx=4)
        tk.Button(toolbar, text="Start", bg=DarkStyle.BTN, fg=DarkStyle.FG, command=self.start, relief="flat").pack(side="left", padx=4)
        tk.Button(toolbar, text="Stop", bg=DarkStyle.BTN, fg=DarkStyle.FG, command=self.stop, relief="flat").pack(side="left", padx=4)

        tk.Checkbutton(toolbar, text="Blur BG", bg=DarkStyle.BG, fg=DarkStyle.FG, selectcolor=DarkStyle.BG,
                       variable=tk.BooleanVar(value=True), command=self.toggle_blur).pack(side="left", padx=10)

        tk.Label(toolbar, text="Neon", fg=DarkStyle.FG, bg=DarkStyle.BG).pack(side="left", padx=6)
        self.neon_scale = tk.Scale(toolbar, from_=0.2, to=2.0, resolution=0.1, orient="horizontal",
                                   bg=DarkStyle.BG, fg=DarkStyle.FG, troughcolor=DarkStyle.BTN, command=self.on_neon)
        self.neon_scale.set(self.neon_intensity)
        self.neon_scale.pack(side="left", padx=4)

        tk.Label(toolbar, text="Trail", fg=DarkStyle.FG, bg=DarkStyle.BG).pack(side="left", padx=6)
        self.trail_scale = tk.Scale(toolbar, from_=1, to=20, orient="horizontal", bg=DarkStyle.BG,
                                    fg=DarkStyle.FG, troughcolor=DarkStyle.BTN, command=self.on_trail)
        self.trail_scale.set(self.trail_length)
        self.trail_scale.pack(side="left", padx=4)

        self.record_var = tk.BooleanVar(value=False)
        tk.Checkbutton(toolbar, text="Record", bg=DarkStyle.BG, fg=DarkStyle.FG, variable=self.record_var,
                       command=self.toggle_record, selectcolor=DarkStyle.BG).pack(side="left", padx=10)
        self.filename_entry = tk.Entry(toolbar, width=18, bg=DarkStyle.BTN, fg=DarkStyle.FG, insertbackground=DarkStyle.FG)
        self.filename_entry.insert(0, "recording_clone.avi")
        self.filename_entry.pack(side="left", padx=4)

        if MATPLOTLIB_AVAILABLE:
            self.plot_btn = tk.Checkbutton(toolbar, text="3D", bg=DarkStyle.BG, fg=DarkStyle.FG, command=self.toggle_3d)
            self.plot_btn.pack(side="left", padx=8)

        tk.Button(toolbar, text="Open Folder", bg=DarkStyle.BTN, fg=DarkStyle.FG, command=self.open_folder, relief="flat").pack(side="right", padx=6)

    def build_video_area(self):
        # large canvas area using a Label for image updates
        self.video_label = tk.Label(self.root, bg="black")
        self.video_label.pack(fill="both", expand=True, padx=6, pady=(0,6))

    def build_statusbar(self):
        status = tk.Frame(self.root, bg=DarkStyle.BG)
        status.pack(fill="x", padx=6, pady=(0,6))
        self.fps_label = tk.Label(status, text="FPS: 0", fg=DarkStyle.FG, bg=DarkStyle.BG)
        self.fps_label.pack(side="left")
        self.gesture_label = tk.Label(status, text="Gestures: None", fg=DarkStyle.ACCENT, bg=DarkStyle.BG)
        self.gesture_label.pack(side="left", padx=20)

    # -------------- Actions --------------
    def open_camera(self):
        try:
            cam_idx = int(self.cam_entry.get())
        except Exception:
            cam_idx = 0
            self.cam_entry.delete(0, tk.END); self.cam_entry.insert(0, "0")
        self.camera_index = cam_idx
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = cv2.VideoCapture(self.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Camera", f"Cannot open camera index {self.camera_index}")

    def start(self):
        if not self.cap:
            self.open_camera()
        self.running = True
        self.start_time = time.perf_counter()
        self.frame_count = 0
        self.fps_display = 0
        self.trails = []
        print("Started")

    def stop(self):
        self.running = False
        # stop writer if active
        if self.writer:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None
        print("Stopped")

    def toggle_record(self):
        self.recording = bool(self.record_var.get())
        # writer will be opened lazily on first frame
        if not self.recording and self.writer:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None

    def toggle_blur(self):
        self.use_blur = not self.use_blur

    def toggle_3d(self):
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showinfo("3D", "Matplotlib not available")
            return
        self.use_3d = not self.use_3d

    def on_neon(self, val):
        try:
            self.neon_intensity = float(val)
        except Exception:
            pass

    def on_trail(self, val):
        try:
            self.trail_length = int(val)
        except Exception:
            pass

    def open_folder(self):
        path = os.getcwd()
        if os.name == "nt":
            os.startfile(path)
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    # -------------- Main Loop & Processing --------------
    def update_loop(self):
        """Main scheduled method called by Tk.after"""
        if self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                composed, gestures = self.process_frame(frame)
                # update recording
                if self.recording:
                    if self.writer is None:
                        fname = self.filename_entry.get().strip() or "recording_clone.avi"
                        fourcc = cv2.VideoWriter_fourcc(*"XVID")
                        h, w = composed.shape[:2]
                        try:
                            self.writer = cv2.VideoWriter(fname, fourcc, 20.0, (w, h))
                        except Exception as e:
                            print("VideoWriter error:", e); self.writer = None
                    if self.writer is not None:
                        try:
                            self.writer.write(composed)
                        except Exception:
                            pass
                # update Tk image
                self.show_frame_on_label(composed)
                # update status
                self.gesture_label.config(text=f"Gestures: {', '.join(gestures) if gestures else 'None'}")
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_update > 0.5:
                    dt = now - self.prev_time if self.prev_time else 1e-6
                    self.fps_display = int(1.0 / max(1e-6, dt))
                    self.prev_time = now
                    self.last_fps_update = now
                self.fps_label.config(text=f"FPS: {self.fps_display}")

        # schedule next call
        self.root.after(10, self.update_loop)

    def show_frame_on_label(self, frame_bgr):
        """Convert BGR frame to PhotoImage and show on label."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # resize to label size while keeping aspect ratio (limit width)
            label_w = self.video_label.winfo_width() or 800
            label_h = self.video_label.winfo_height() or 480
            # keep as original if small
            h, w = rgb.shape[:2]
            scale = min(label_w / w, label_h / h, 1.6)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(rgb, (new_w, new_h))
            # encode PNG bytes and feed to Tk PhotoImage
            encoded = cv2.imencode(".png", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))[1].tobytes()
            photo = tk.PhotoImage(data=encoded)
            self.video_label.config(image=photo)
            self.video_label.image = photo
        except Exception as e:
            # fallback: show black
            print("show_frame error:", e)

    def process_frame(self, frame):
        """Main CV processing: pose detection, clone mirror, glow, trails, blur, gestures, 3D."""
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get segmentation mask
        seg_res = SEG.process(frame_rgb)
        mask = None
        if seg_res and seg_res.segmentation_mask is not None:
            mask = np.expand_dims(seg_res.segmentation_mask, axis=2).astype(np.float32)

        pose_res = POSE.process(frame_rgb)
        neon_color = get_gradient_color(time.time() - self.start_time)
        pulse = 0.8 + 0.2 * math.sin((time.time() - self.start_time) * 6)
        neon_color = tuple(int(np.clip(c * self.neon_intensity * pulse, 0, 255)) for c in neon_color)

        glow_layer = np.zeros_like(frame, dtype=np.uint8)
        gestures_detected = []

        if pose_res and pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            # prepare mirrored coordinates
            clone_2d = []
            clone_3d = []
            for point in lm:
                x2 = int((1 - point.x) * w)
                y2 = int(point.y * h)
                clone_2d.append((x2, y2))
                clone_3d.append((1 - point.x, point.y, point.z))
            # draw connections
            for (i, j) in POSE_CONNECTIONS:
                if i < len(clone_2d) and j < len(clone_2d):
                    draw_smooth_line(glow_layer, clone_2d[i], clone_2d[j], neon_color, thickness=6)
            # draw joints and populate trail points
            current_joints = []
            for (x, y) in clone_2d:
                if 0 <= x < w and 0 <= y < h:
                    current_joints.append((x, y))
                    draw_smooth_circle(glow_layer, (x, y), radius=7, color=neon_color)

            self.trails.append(current_joints)
            if len(self.trails) > self.trail_length:
                self.trails.pop(0)

            # draw trails (older fainter)
            L = len(self.trails)
            for t_idx, tframe in enumerate(self.trails):
                alpha = ((t_idx + 1) / max(1, L)) * 0.45
                trail_color = tuple(int(c * alpha) for c in neon_color)
                for (x, y) in tframe:
                    cv2.circle(glow_layer, (x, y), 3, trail_color, -1, lineType=cv2.LINE_AA)

            # gestures:
            gest = simple_gestures(lm)
            for k, v in gest.items():
                if v:
                    gestures_detected.append(k)

            # 3D plotting update (every 8 frames)
            if self.use_3d and MATPLOTLIB_AVAILABLE and (self.frame_count % 8 == 0):
                try:
                    self.ax.clear()
                    self.ax.set_xlim([-1,1]); self.ax.set_ylim([-1,1]); self.ax.set_zlim([-1,1])
                    self.ax.set_facecolor("black")
                    line_color = tuple(c/255 for c in neon_color)
                    for (i, j) in POSE_CONNECTIONS:
                        if i < len(clone_3d) and j < len(clone_3d):
                            xs = [clone_3d[i][0], clone_3d[j][0]]
                            ys = [clone_3d[i][1], clone_3d[j][1]]
                            zs = [clone_3d[i][2], clone_3d[j][2]]
                            self.ax.plot(xs, ys, zs, color=line_color, linewidth=2, alpha=0.9)
                    xs, ys, zs = zip(*clone_3d)
                    self.ax.scatter(xs, ys, zs, c=[line_color], s=30, alpha=0.9)
                    plt.draw(); plt.pause(0.001)
                except Exception as e:
                    self.use_3d = False

        else:
            # show scanning text on frame if no detection
            txt = "SCANNING..."
            tsize = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            tx = (w - tsize[0]) // 2
            ty = (h + tsize[1]) // 2
            cv2.putText(frame, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.0, get_gradient_color(time.time()-self.start_time), 2, cv2.LINE_AA)

        # Apply glow blur and overlay
        blurred = cv2.GaussianBlur(glow_layer, (0,0), sigmaX=9, sigmaY=9)
        composed = cv2.addWeighted(frame, 0.7, blurred, 0.9, 0)

        # apply background blur if enabled and we have a mask
        if self.use_blur and mask is not None:
            composed = apply_background_blur(composed, mask, blur_strength=21)

        return composed, gestures_detected

    def on_close(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        if self.writer:
            try:
                self.writer.release()
            except Exception:
                pass
        try:
            POSE.close()
            SEG.close()
        except Exception:
            pass
        self.root.destroy()

# ------------------ Run App ------------------
if __name__ == "__main__":
    root = tk.Tk()
    # set minimum size to look nice
    root.geometry("1000x700")
    app = CloneTrackerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
