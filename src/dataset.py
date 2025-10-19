import os
import glob
import cv2
from torch.utils.data import Dataset

class CelebVDataset(Dataset):
    def __init__(self, root_dir="data/celebv-text", seq_len=32):
        print(f"[INFO] Initializing CelebVDataset with root_dir={root_dir} and seq_len={seq_len}")
        self.video_dir = os.path.join(root_dir, "video")
        self.text_dirs = {cat: os.path.join(root_dir, cat)
                          for cat in ["action", "emotion", "face_details", "light_direction",
                                      "light_intensity", "light_color_temp"]}
        self.video_files = sorted(glob.glob(os.path.join(self.video_dir, "*.mp4")))
        print(f"[INFO] Found {len(self.video_files)} video files")
        self.seq_len = seq_len

    def _read_text(self, base):
        full_text = []
        for cat, dir_path in self.text_dirs.items():
            txt_path = os.path.join(dir_path, f"{base}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    lines = f.readlines()
                    full_text.append(lines[0].strip())
            else:
                print(f"[WARNING] Text file not found: {txt_path}")
        return " ".join(full_text)

    def _extract_frames(self, video_path, target_size=(224, 224)):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)  # resize to fixed size
            frames.append(frame)
            frame_count += 1
            if len(frames) >= self.seq_len:
                break
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        base = os.path.splitext(os.path.basename(video_path))[0]
        text = self._read_text(base)
        frames = self._extract_frames(video_path)
        return {"text": text, "frames": frames}

    def __len__(self):
        return len(self.video_files)
