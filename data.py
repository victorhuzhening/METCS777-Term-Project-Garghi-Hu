import cv2
import os
import pyarrow as pa
import pyarrow.csv as csv
import mediapipe as mp
from torch.utils.data import Dataset
from utils import *


def load_sentence_labels(csv_path):
    """
    Helper to create labels lookup dict.
    Reads from the csv:
        ["sentence_id", "sentence"]

    Returns:
        SENTENCE_LABELS: {
            "video_id": sentence label
        }
    """
    parse_options = csv.ParseOptions(delimiter="\t")

    convert_options = csv.ConvertOptions(
        column_types={
            "SENTENCE_NAME": pa.string(),
            "SENTENCE": pa.string(),
        },
        include_columns=["SENTENCE_NAME", "SENTENCE"],
    )

    table = csv.read_csv(csv_path, parse_options=parse_options, convert_options=convert_options)

    sentence_id = table["SENTENCE_NAME"].to_pylist()
    sentence = table["SENTENCE"].to_pylist()
    return dict(zip(sentence_id, sentence))


class How2Sign(Dataset):
    def __init__(self, video_dir, MP_model, body_cfg, body_model, labels_path, output_dir=None):
        self.video_dir = video_dir
        self.MP_model = MP_model
        self.body_cfg = body_cfg
        self.body_model = body_model
        self.labels_path = labels_path
        self.output_dir = str(output_dir) if output_dir is not None else None

        self.video_paths = [os.path.join(video_dir, path) for path in os.listdir(video_dir)]
        self.sentence_labels = load_sentence_labels(self.labels_path)

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
        hand_results_sequence = []      # list[HandLandmarkerResult]
        pose_results_sequence = []     # list[list[PoseDataSample]]

        for frame_rgb in self._iter_video_as_frames(video_path):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=frame_rgb)

            hand_landmarks = self.MP_model.detect(mp_image)
            pose_data = self.body_cfg.topdown_infer(self.body_model, frame_rgb)

            hand_results_sequence.append(hand_landmarks)
            pose_results_sequence.append(pose_data)

            hand_json = hand_landmarks_to_json(hand_results_sequence)
            pose_json = pose_data_to_json(pose_results_sequence)

        return {
            "hand_landmarks": hand_json,
            "body_info": pose_json,
        }

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        file_name_ext = os.path.basename(video_path)
        file_name, _ = os.path.splitext(file_name_ext)

        inference_data = self.extract_information(video_path)
        sample = {
            'filename': str(video_path),
            'hand_landmarks': inference_data['hand_landmarks'],
            'body_coordinates': inference_data['body_info'],
            'label': self.sentence_labels[file_name]
        }
        return sample


class CameraCfg:
    """
    Camera configuration to set up a livestream input using webcam
    Output is defined using VideoWriter
    """

    def __init__(self, cameraIdx: int, fps: float, is_array: bool):
        self.CameraIdx = cameraIdx  # default camera (usually webcam)
        self.FPS = fps
        self.OutputPath = "output.mp4" if not is_array else "coordinates.csv"

    def create_camera(self):
        cam = cv2.VideoCapture(self.CameraIdx, cv2.CAP_DSHOW)
        if not cam.isOpened():
            raise RuntimeError("ERROR: could not open camera")
        frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # define codec
        out = cv2.VideoWriter(self.OutputPath, fourcc, self.FPS, (frame_width, frame_height))
        return cam, out, frame_width, frame_height


class CameraData(Dataset):
    def __init__(self, camera_idx, MP_model, body_cfg, body_model):
        self.camera_idx = camera_idx
        self.MP_model = MP_model
        self.body_cfg = body_cfg
        self.body_model = body_model
