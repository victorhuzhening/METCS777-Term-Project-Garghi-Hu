import cv2
import os
import mediapipe as mp
from torch.utils.data import Dataset

from mediapipe.tasks.python import vision
from torch.utils.data import Dataset



class How2Sign(Dataset):
    def __init__(self, video_dir, MP_model, body_cfg, body_model, output_dir = None):
        self.video_dir = video_dir
        self.MP_model = MP_model
        self.body_cfg = body_cfg
        self.body_model = body_model
        self.output_dir = str(output_dir) if output_dir is not None else None

        self.video_paths = [os.path.join(video_dir, path) for path in os.listdir(video_dir)]


    def __len__(self):
        return len(self.video_paths)


    def _iter_video_as_frames(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError("ERROR: Cannot open video file")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert cv frame (BGR) to RGB
                yield frame_rgb
        finally:
            cap.release()


    def extract_information(self, video_path):
        results = {"hands": [], "body": []}
        for frame_rgb in self._iter_video_as_frames(video_path):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=frame_rgb)

            hand_landmarks = self.MP_model.detect(mp_image)
            pose_data = self.body_cfg.topdown_infer(self.body_model, frame_rgb)

            results["hands"].append(hand_landmarks)
            results["body"].append(pose_data)

        return results


    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        inference_data = self.extract_information(video_path)
        sample = {
            'filename': str(video_path),
            'landmark_coordinates': inference_data['hands'],
            'body_coordinates': inference_data['body']
        }
        return sample