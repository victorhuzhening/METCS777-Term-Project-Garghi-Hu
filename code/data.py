import os
from glob import glob
import pyarrow as pa
import pyarrow.csv as csv
import mediapipe as mp
from torch.utils.data import Dataset
from utils import *
from tokenizer import *
from transforms import *


class CameraCfg:
    """
    Camera configuration to set up a livestream input using webcam.
    Function used for demo and livestream inference - currently unused :(
    Output is defined using VideoWriter.
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



def extract_coordinate_sequences_for_video(
    video_path: str,
    MP_model,
    body_cfg,
    body_model,
    num_keypoints: int = 17,
):
    """
    Runs MediaPipe and MMPose models over all frames and returns
    hand/body coordinate sequences. Logic taken from extract_coordinate_sequences because
    this function needs to be used for inference.
    return:
        hand_seq: dict
        body_seq: dict
    """
    hand_results = []
    body_results = []

    for frame_rgb in iter_video_as_frames(video_path):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=frame_rgb)
        hand_coordinates = MP_model.detect(mp_image)
        pose_coordinates = body_cfg.topdown_infer(body_model, frame_rgb)

        hand_results.append(hand_coordinates)
        body_results.append(pose_coordinates)

    hand_seq = hand_coordinates_to_seq(hand_results)
    body_seq = pose_coordinates_to_seq(body_results, num_keypoints=num_keypoints)
    return hand_seq, body_seq


def build_feature_tensor_from_sequences(
    hand_seq,
    body_seq,
    frame_subsample: int = 1,
    max_frames: int | None = None,
):
    """
    Build a [T, D] FloatTensor feature tensor from hand and body coordinate sequences, where T is frame dimension,
    and D is number of features.
    Returns ML-friendly PyTorch FloatTensor arrays.
    This function is also used for inference.
    """
    hand_frame_dim = hand_seq.get("num_frames", len(hand_seq.get("frames", [])))
    body_frame_dim = body_seq.get("num_frames", len(body_seq.get("frames", [])))
    frame_dim = min(hand_frame_dim, body_frame_dim)  # match frames by min number
    if frame_dim == 0:
        raise RuntimeError("Video produced 0 frames while building feature tensor.")

    frames_hand = hand_seq["frames"]
    frames_body = body_seq["frames"]

    frame_subsample = max(1, int(frame_subsample))
    frame_indices = list(range(0, frame_dim, frame_subsample))
    if max_frames:
        frame_indices = frame_indices[: max_frames]

    # Temporal transformations
    frame_indices = temporal_jitter_and_shuffle(
        frame_indices,
        num_frames=frame_dim,
        max_jitter=1,
        jitter_prob=0.2,
        shuffle_prob=0.15,
    )

    feature_seq = []

    for idx in frame_indices:
        hand_frame = frames_hand[idx]
        body_frame = frames_body[idx]

        left_hand = hand_frame["left_hand"]    # [21,3]
        right_hand = hand_frame["right_hand"]  # [21,3]

        # Edge case safety check because transforms require torch tensors
        if not isinstance(left_hand, torch.Tensor):
            left_hand = torch.tensor(left_hand, dtype=torch.float32)
        if not isinstance(right_hand, torch.Tensor):
            right_hand = torch.tensor(right_hand, dtype=torch.float32)

        # Spatial transforms (require torch tensors)
        left_hand = random_affine_transforms(left_hand)
        right_hand = random_affine_transforms(right_hand)

        left_hand = left_hand.reshape(-1)  # flatten to 1D [21*3]
        right_hand = right_hand.reshape(-1)

        body_coordinates = body_frame["body_coordinates"]  # [K,2]
        body_scores = body_frame["body_scores"]            # [K]

        if not isinstance(body_coordinates, torch.Tensor):
            body_coordinates = torch.tensor(body_coordinates, dtype=torch.float32)
        if not isinstance(body_scores, torch.Tensor):
            body_scores = torch.tensor(body_scores, dtype=torch.float32)

        body_coordinates = random_affine_transforms(body_coordinates)
        body_coordinates = body_coordinates.reshape(-1)    # flatten to 1D [2K]

        feature_vector = torch.concat(
            (left_hand, right_hand, body_coordinates, body_scores),
            dim=0,
        )
        feature_seq.append(feature_vector)

    if not feature_seq:
        raise RuntimeError("No features extracted while building feature tensor.")

    feature_seq_tensor = torch.stack(feature_seq, dim=0).float()  # [T', D]
    return feature_seq_tensor


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

        # vocab + tokenizer block
        if vocab is None:
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
        """
        return extract_coordinate_sequences_for_video(
            video_path=video_path,
            MP_model=self.MP_model,
            body_cfg=self.body_cfg,
            body_model=self.body_model,
            num_keypoints=self.num_keypoints,
        )


    def build_feature_tensor(self, hand_seq, body_seq):
        """
        Build a [T, D] FloatTensor feature tensor from hand and body coordinate sequences.
        """
        feature_seq_tensor = build_feature_tensor_from_sequences(
            hand_seq=hand_seq,
            body_seq=body_seq,
            frame_subsample=self.frame_subsample,
            max_frames=self.max_frames,
        )
        return feature_seq_tensor

    def __getitem__(self, idx):
        """
        Returns a dict ready for model training/prediction:
        """
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
            "features": feature_tensor,    # [T', D]
            "feature_len": tensor_len,     #  int
            "label_ids": label_ids,        # [L]
            "label_len": label_len,        #  int
            "filename": str(video_path),   # str
            "raw_label": label_text,       # str
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
        "feature_len": feature_len_tensor,  # [B]
        "labels": label_batch,              # [B, max_L]
        "label_len": label_len_tensor,      # [B]
        "filenames": filenames,             # str
        "raw_labels": raw_labels,           # str
    }


class PrecomputedASLData(Dataset):
    """
    Loads precomputed ASL samples from feature directory.
    A separate vocab meta file must exist in the same directory.
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