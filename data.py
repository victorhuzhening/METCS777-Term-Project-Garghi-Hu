import cv2
import os
import mediapipe as mp
from torch.utils.data import Dataset

from mediapipe.tasks.python import vision
from torch.utils.data import Dataset

class How2SignHands(Dataset):
    def __init__(self, video_dir, MP_model, output_dir = None):
        self.video_dir = video_dir
        self.MP_model = MP_model
        self.output_dir = str(output_dir) if output_dir is not None else None

        self.video_paths = [os.path.join(video_dir, path) for path in os.listdir(video_dir)]

    def __len__(self):
        return len(self.video_paths)

    def _iter_frames_as_mp_images(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError("ERROR: Cannot open video file")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert cv frame (BGR) to RGB
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=frame_rgb)
                yield mp_image
        finally:
            cap.release()


    def _get_landmarks(self, video_path):
        results = []
        for mp_image in self._iter_frames_as_mp_images(video_path):
            coordinates = self.MP_model.detect(mp_image)
            results.append(coordinates)
        return results

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        landmark_coordinates = self._get_landmarks(video_path)
        sample = {
            'filename': str(video_path),
            'landmark_coordinates': landmark_coordinates
        }
        return sample