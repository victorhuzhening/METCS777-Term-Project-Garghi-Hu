import cv2
import os
from glob import glob
import torch
import pyarrow as pa
import pyarrow.csv as csv
import mediapipe as mp
from torch.utils.data import Dataset
from utils import *
from tokenizer import *


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


class ASLData(Dataset):
    def __init__(self,
                 video_dir,
                 MP_model,
                 body_cfg,
                 body_model,
                 labels_path,
                 max_frames,
                 min_frequency = 1,
                 vocab = None,
                 frame_subsample = 1,
                 num_keypoints = 17,
                 output_dir=None):
        super().__init__()
        self.video_dir = video_dir
        self.MP_model = MP_model
        self.body_cfg = body_cfg
        self.body_model = body_model
        self.labels_path = labels_path
        self.max_frames = max_frames
        self.vocab = vocab
        self.frame_subsample = max(1, int(frame_subsample))
        self.num_keypoints = num_keypoints
        self.output_dir = str(output_dir) if output_dir is not None else None

        self.video_paths = [os.path.join(video_dir, path) for path in os.listdir(video_dir)]
        self.sentence_labels = load_sentence_labels(self.labels_path)

        # ----- vocab + tokenizer -----
        if vocab is None:
            # build vocab from all labels in the TSV
            sentences = self.sentence_labels.values()
            self.vocab = build_vocab_from_sentences(sentences, min_freq=min_frequency)
        else:
            self.vocab = vocab

        self.pad_id = self.vocab["<pad>"]
        self.unk_id = self.vocab["<unk>"]

        def _tokenizer_fn(text):
            tokens = ["<bos>"] + basic_tokenize(text) + ["<eos>"]
            return [self.vocab.get(tok, self.unk_id) for tok in tokens]

        self.tokenizer_fn = _tokenizer_fn

    def __len__(self):
        return len(self.video_paths)

    def extract_coordinate_sequences(self, video_path):
        """
        Run Hand Landmarker and MMPose models over all frames and get raw sequences.

        :param video_path:
        :return:
            hand_seq: dict
            body_seq: dict
        """
        hand_results = []
        body_results = []

        for frame_rgb in iter_video_as_frames(video_path):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=frame_rgb)
            hand_coordinates = self.MP_model.detect(mp_image)
            pose_coordinates = self.body_cfg.topdown_infer(self.body_model, frame_rgb)

            hand_results.append(hand_coordinates)
            body_results.append(pose_coordinates)

        hand_seq = hand_coordinates_to_seq(hand_results)
        body_seq = pose_coordinates_to_seq(body_results, num_keypoints=self.num_keypoints)
        return hand_seq, body_seq

    def build_feature_tensor(self, hand_seq, body_seq):
        """
        Build a [T, D] FloatTensor feature tensor from hand and body coordinate sequences, where T is frame dimension,
        and D is number of features.
        Returns ML-friendly PyTorch FloatTensor arrays.
        """
        hand_frame_dim = hand_seq.get("num_frames", len(hand_seq.get("frames", [])))
        body_frame_dim = body_seq.get("num_frames", len(body_seq.get("frames", [])))
        frame_dim = min(hand_frame_dim, body_frame_dim) # match frames by min number
        if frame_dim == 0:
            raise RuntimeError("Video produced 0 frames while building feature tensor.")

        frames_hand = hand_seq["frames"]
        frames_body = body_seq["frames"]

        frame_indices = list(range(0, frame_dim, self.frame_subsample))
        if self.max_frames:
            frame_indices = frame_indices[: self.max_frames]

        feature_seq = []

        for idx in frame_indices:
            hand_frame = frames_hand[idx]
            body_frame = frames_body[idx]

            left_hand = np.array(hand_frame["left_hand"], dtype=np.float32)  # [21, 3]
            right_hand = np.array(hand_frame["right_hand"], dtype=np.float32)

            left_hand = left_hand.reshape(-1) # flatten to 1D
            right_hand = right_hand.reshape(-1)

            body_coordinates = np.array(body_frame["body_coordinates"], dtype=np.float32)  # [2K]
            body_scores = np.array(body_frame["body_scores"], dtype=np.float32)  # [K]

            feature_vector = np.concatenate([
                left_hand, right_hand, body_coordinates, body_scores],
                axis=0
            )
            feature_seq.append(feature_vector)

        if not feature_seq:
            raise RuntimeError("No features extracted while building feature tensor.")

        feature_seq_stack = np.stack(feature_seq, axis=0)
        feature_seq_tensor = torch.from_numpy(feature_seq_stack).float()
        return feature_seq_tensor

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        file_name_ext = os.path.basename(video_path)
        file_name, _ = os.path.splitext(file_name_ext)

        label_text = self.sentence_labels[file_name]
        hand_seq, body_seq = self.extract_coordinate_sequences(video_path)
        feature_tensor = self.build_feature_tensor(hand_seq, body_seq)
        tensor_len = len(feature_tensor)

        label_ids_list = self.tokenizer_fn(label_text)
        label_ids = torch.tensor(label_ids_list, dtype=torch.long)
        label_len = len(label_ids_list)

        return {
            "features": feature_tensor,  # [T', D]
            "feature_len": tensor_len,  #  int
            "label_ids": label_ids,  # [L]
            "label_len": label_len,  #  int
            "filename": str(video_path), # str
            "raw_label": label_text, # str
        }


def asl_collate_func(batch, pad_id):
    """
    Collate function for ASLData dataset.
    Pads sequences in time dimension, and label sequences in length, using pad_id for text.
    """
    batch_size = len(batch)

    feature_len = [b["feature_len"] for b in batch]
    label_len = [b["label_len"] for b in batch]

    max_feature_len = max(feature_len)
    max_label_len = max(label_len)

    feature_dim = batch[0]["features"].shape[1]

    feature_batch = torch.zeros(batch_size, max_feature_len, feature_dim, dtype=torch.float32)
    label_batch = torch.full(
        (batch_size, max_label_len), fill_value=pad_id, dtype=torch.long # use long for int
    )

    feature_len_tensor = torch.tensor(feature_len, dtype=torch.long)
    label_len_tensor = torch.tensor(label_len, dtype=torch.long)

    filenames = []
    raw_labels = []

    for i, sample in enumerate(batch):
        sample_feature_len = sample["feature_len"]
        sample_label_len = sample["label_len"]

        feature_batch[i, :sample_feature_len] = sample["features"]
        label_batch[i, :sample_label_len] = sample["label_ids"]

        filenames.append(sample["filename"])
        raw_labels.append(sample["raw_label"])

    return {
        "features": feature_batch,          # [B, max_T, D]
        "feature_len": feature_len_tensor, # [B]
        "labels": label_batch,          # [B, max_L]
        "label_len": label_len_tensor,  # [B]
        "filenames": filenames,         # str
        "raw_labels": raw_labels,       # str
    }


class PrecomputedASLData(Dataset):
    """
    Loads precomputed ASL samples from .pt files.

    Each sample_XXXXX.pt must be a dict with keys:
      - pose:       FloatTensor [T, D]
      - pose_len:   int
      - label_ids:  LongTensor [L]
      - label_len:  int
      - filename:   str
      - raw_label:  str

    A separate vocab_meta.pt must exist in the same directory with:
      {
        "vocab": {token: id},
        "pad_id": int,
      }
    """

    def __init__(self, data_dir: str):
        super().__init__()
        self.sample_paths = sorted(glob(os.path.join(data_dir, "sample_*.pt")))
        if not self.sample_paths:
            raise RuntimeError(f"No sample_*.pt files found in {data_dir}")

        vocab_meta = torch.load(os.path.join(data_dir, "vocab_meta.pt"))
        self.vocab = vocab_meta["vocab"]
        self.pad_id = vocab_meta["pad_id"]

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        sample = torch.load(self.sample_paths[idx], map_location="cpu")
        # Safety check to satisfy collate function expectations
        return {
            "features": sample["features"],
            "feature_len": sample["feature_len"],
            "label_ids": sample["label_ids"],
            "label_len": sample["label_len"],
            "filename": sample["filename"],
            "raw_label": sample["raw_label"],
        }