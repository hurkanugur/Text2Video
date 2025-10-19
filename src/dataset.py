import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from src import config

class CelebVDataset(Dataset):
    """Simple CelebV dataset loader."""

    # --------------------- Folders ---------------------

    VIDEO_FOLDER = "video"
    TEXT_FOLDERS = [
        "action", "emotion", "face_details",
        "light_direction", "light_intensity", "light_color_temp"
    ]

    # --------------------- Initialization ---------------------

    def __init__(self):
        print(f"[INFO] Initializing dataset with root_dir={config.DATASET_PATH}, frames_per_video={config.FRAMES_PER_VIDEO}")

        self.video_directory_path = os.path.join(config.DATASET_PATH, self.VIDEO_FOLDER)
        self.text_directory_paths = {cat: os.path.join(config.DATASET_PATH, cat) for cat in self.TEXT_FOLDERS}
        self.video_file_paths = self._get_video_file_paths()
        self._print_text_file_stats()

    # --------------------- Initialization helpers ---------------------

    def _get_video_file_paths(self):
        """Get sorted list of video file paths."""
        video_file_paths = sorted(glob.glob(os.path.join(self.video_directory_path, "*.mp4")))
        print(f"[INFO] Found {len(video_file_paths)} video files in {self.VIDEO_FOLDER}")
        return video_file_paths

    def _print_text_file_stats(self):
        """Print how many text files exist per category."""
        for category, folder in self.text_directory_paths.items():
            count = len(glob.glob(os.path.join(folder, "*.txt")))
            print(f"[INFO] Found {count} text files in {category}")

    # --------------------- Data loading helpers ---------------------

    def _read_text(self, video_name):
        """
        Read all lines from all text files for a given video and concatenate them.
        
        Returns:
            A single string combining all categories and all lines per category.
        """
        texts = []
        for category, folder in self.text_directory_paths.items():
            txt_path = os.path.join(folder, f"{video_name}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    # Read all lines, strip whitespace, ignore empty lines
                    lines = [line.strip() for line in f if line.strip()]
                    texts.append(" ".join(lines))
            else:
                print(f"[WARNING] Missing text file: {txt_path}")
        return " ".join(texts)

    def _extract_frames(self, video_path):
        """
        Extract the first [config.FRAMES_PER_VIDEO] frames from a video and resize them to 224x224.

        Args:
            video_path (str): Path to the video file.

        Returns:
            np.ndarray: A NumPy array of shape 
                (num_frames, 224, 224, 3), where num_frames ≤ config.FRAMES_PER_VIDEO, 
                representing the extracted RGB frames.

        Raises:
            ValueError: If no frames could be read from the video.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while len(frames) < config.FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No frames found in video {video_path}")

        return np.stack(frames)

    # --------------------- Dataset interface ---------------------

    def __getitem__(self, idx):
        video_path = self.video_file_paths[idx]
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        text = self._read_text(video_name)
        frames = self._extract_frames(video_path)
        return {"text": text, "frames": frames}

    def __len__(self):
        return len(self.video_file_paths)
